from copy import deepcopy
from uuid import uuid4
from warnings import warn

import numpy as np
from lineartree import LinearTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import _check_sample_weight

CONCAVE = 'concave'
CONVEX = 'convex'
MIN = 'concave'
MAX = 'convex'

LOWER_BETTER = 'lower'
HIGHER_BETTER = 'higher'

_DIRECTIONS = (CONCAVE, CONVEX, MIN, MAX)
_SCORE_MODES = (LOWER_BETTER, HIGHER_BETTER)

_extremal_fn = {
    'convex': (np.max, np.argmax),
    'concave': (np.min, np.argmin),
}

IT_LIM_HIT_MSG = """
The emergency iteration limit was hit. 
The regressor did not naturally finish the fitting, but was interrupted.
This could mean that the iteration limit must be increased (but keep in mind that
more and more hyperplanes are generated!), that the direction (concave/convex) is
wrong, that the data is particularly problematic, etc.
Use this estimator with caution.  
"""


FEW_SAMPLES_MSG = """
There are very few samples.
Minimum samples per leaf is set at {min_leaf_samples}.
The number of training samples that were passed is {nsamples}.
In order to generate the first split, double the minimum is required ({nsamples2}). 
"""


def _leaves(linear_tree_regr):
    s = linear_tree_regr.summary()
    leaves = [k for k, v in s.items() if 'th' not in v.keys()]
    return leaves


def _dump_models(linear_tree_regr, X, y, sw):
    s = linear_tree_regr.summary()
    leaves = _leaves(linear_tree_regr)
    selection = linear_tree_regr.apply(X)
    model_names = {l: uuid4().hex for l in leaves}
    models = {model_names[l]: s[l]['models'] for l in leaves}
    data_split = {model_names[l]:
                        (X[selection == l, :],
                         y[selection == l],
                         sw[selection == l])
                    for l in leaves}
    return models, data_split


def _subsets_to_freeze(old_cap, new_cap, X, y, sw, min_leaf, old_frozen, data_split):
    # this function checks if some hyperplanes have fewer points than the minimum allowed
    # following a split (and possibly a refit), and if so it saves their last
    # valid dataset in a "frozen subsets" dictionary that is used when the model is refit.

    old_ids = set(old_cap.hyperplanes_ids_)
    new_ids = set(new_cap.hyperplanes_ids_)

    deleted_hp = old_ids - new_ids
    introduced_ids = new_ids - old_ids
    assert len(deleted_hp) == 1
    deleted_hp = list(deleted_hp)[0]

    old_ids.remove(deleted_hp)

    old_part_x, old_part_y, old_part_sw = old_cap.partition_data(X, y)
    new_part_x, new_part_y, new_part_sw = new_cap.partition_data(X, y)

    frozen_subsets = {}

    # check the old hyperplanes
    for part_id in old_ids:
        if new_part_x[part_id].shape[0] >= min_leaf:
            # this partition has no problems
            continue
        elif part_id in old_frozen.keys():
            # if the partition is underweight but was already frozen,
            # just carry on the frozen subset
            frozen_subsets[part_id] = old_frozen[part_id]
        else:
            # this hyperplane became underweight after the split so we save its data
            frozen_subsets[part_id] = (old_part_x[part_id], old_part_y[part_id], old_part_sw[part_id])

    # the introduced planes might be underweight as well. if so, they are
    # assigned the subset of points they were given by the splitter.
    # this happens only on a successful lintree round whose new hyperplanes
    # happen to have points "stolen" by old hyperplanes; if a combinatorial round was
    # performed, the introduced planes must not be underweight, because underweight
    # splits are discarded.
    for part_id in introduced_ids:
        if data_split is None:
            assert new_part_x[part_id].shape[0] > min_leaf
        else:
            if new_part_x[part_id].shape[0] < min_leaf:
                frozen_subsets[part_id] = data_split[part_id]

    return frozen_subsets


class _BaseCAPRegressor(RegressorMixin, BaseEstimator):
    underlying_estimator = LinearRegression(fit_intercept=True)

    def __init__(
            self,
            concavity='convex',
            min_leaf_samples=3,
            attempt_full_cap_on_stop=False,
            save_model_sequence=False,
            score_fn=(mean_squared_error, 'lower'),
            refit_rounds=1,
            emergency_iteration_limit=1000,
    ):
        super().__init__()
        # coherence checks
        if concavity not in _DIRECTIONS:
            raise ValueError(f"Invalid concavity {concavity} "
                             f"({'/'.join(_extremal_fn.keys())})")
        if not (isinstance(score_fn, tuple) and len(score_fn) == 2):
            raise ValueError(f"Invalid score tuple. Pass a tuple "
                             f"(scoring_fn, '{LOWER_BETTER}'/'{HIGHER_BETTER}')")
        if score_fn[1] not in _SCORE_MODES:
            raise ValueError(f"Invalid score direction {score_fn[1]}")
        if emergency_iteration_limit < 0:
            raise ValueError(f"Invalid emergency iteration limit. It must be"
                             f"a positive integer, or 0 for no limit (CAUTION!)")

        self.concavity = concavity
        self.score_fn = score_fn
        self.save_model_sequence = save_model_sequence
        self.min_leaf_samples = int(min_leaf_samples)
        self.refit_rounds = refit_rounds
        self.emergency_iteration_limit = int(emergency_iteration_limit)

    def _more_tags(self):
        return {'poor_score': True,
                'non_deterministic': False,
                'requires_fit': True}

    def _initial_fit(self, X, y, sample_weight=None):
        """Fits a single regressor on the data and sets it
        as the first hyperplane"""
        est = deepcopy(self.underlying_estimator)
        est.fit(X, y, sample_weight=sample_weight)
        self.hyperplanes_.append(est)
        self.hyperplanes_ids_.append(uuid4().hex)

    def _compare_score(self, score, baseline):
        """Compares a score wrt a baseline value, with the correct direction
        (lower/higher better) according to self.score_fn"""

        if self.score_fn[1] == LOWER_BETTER:
            return score < baseline
        elif self.score_fn[1] == HIGHER_BETTER:
            return score > baseline
        else:
            raise ValueError(f"Score mode {self.score_fn[1]} is not valid. "
                             f"It should be one of '{LOWER_BETTER}'/'{HIGHER_BETTER}'")

    def _generate_iterations_number(self):
        if self.emergency_iteration_limit == 0:
            nit = 0
            while True:
                yield nit
                nit += 1
        elif self.emergency_iteration_limit > 0:
            for nit in range(self.emergency_iteration_limit):
                yield nit
        else:
            raise ValueError('Invalid iteration limit, please check.')

    @staticmethod
    def _check_constraint_satisfaction(capregressor, X, y, stolen):
        test_part_x, test_part_y, _ = capregressor.partition_data(X, y)
        samples_constraint_satisfied = all([x.shape[0] >= capregressor.min_leaf_samples or pno in stolen for pno, x in test_part_x.items()])
        return samples_constraint_satisfied

    def purge_dead_hyperplanes(self, X, y):
        part_x, part_y, _ = self.partition_data(X, y)
        dead_hyperplanes = [pno for pno, x in part_x.items() if x.shape[0] == 0]
        for pno in dead_hyperplanes:
            warn(f'Deleted dead hyperplane {pno}')
            kill_index = self.hyperplanes_ids_.index(pno)
            del self.hyperplanes_[kill_index]
            del self.hyperplanes_ids_[kill_index]

    def predict(self, X):
        if not hasattr(self, 'hyperplanes_'):
            raise NotFittedError

        preds = np.stack([m.predict(X) for m in self.hyperplanes_])
        prediction = _extremal_fn[self.concavity][0](preds, axis=0)
        return prediction

    def model_sequence(self):
        """Returns the sequence of models generated by fitting"""
        if not self.model_sequence_:
            raise NotFittedError
        else:
            return self.model_sequence_

    def hyperplanes(self):
        """Returns the hyperplanes A + Bx
           A.shape = (|hyperplanes|,)
           B.shape = (|hyperplanes|, features)
        """
        B = np.stack([m.coef_ for m in self.hyperplanes_])
        A = np.asarray([m.intercept_ for m in self.hyperplanes_])
        return B, A

    def partition_data(self, X, y, sample_weight=None):
        """Divides X, y according to the partition induced by
        the current hyperplanes.
        Returns dicts {part_no: part_X}, {part_no: part_Y}"""

        hp_predictions = np.stack([m.predict(X) for m in self.hyperplanes_])
        partition_assignment = _extremal_fn[self.concavity][1](hp_predictions, axis=0)
        xpart = {}
        ypart = {}
        swpart = {}

        for p_index, p in enumerate(self.hyperplanes_ids_):
            indexing_p = np.atleast_1d(np.squeeze(partition_assignment == p_index))
            xpart[p] = X[indexing_p, :]
            ypart[p] = y[indexing_p]
            if sample_weight is not None:
                swpart[p] = sample_weight[indexing_p]
            else:
                swpart[p] = None

        return xpart, ypart, swpart

    def refit_current_hyperplanes(self, X, y, stolen, sample_weight=None):
        """Refits new hyperplanes in place according to the partitions induced by
        the current hyperplanes."""

        part_x, part_y, part_sw = self.partition_data(X, y, sample_weight)
        new_hyperplanes = []
        for pno, Xp in part_x.items():
            yp = part_y[pno]
            sw = part_sw[pno] if sample_weight is not None else None
            if len(yp) < self.min_leaf_samples:
                Xp, yp, sw = stolen[pno]
            sm = deepcopy(self.underlying_estimator)
            sm.fit(Xp, yp, sample_weight=sw)
            new_hyperplanes.append(sm)

        # verify that the refit model does not violate the min samples per leaf constraint
        # otherwise skip the refit, as per the paper
        test_copy = deepcopy(self)
        test_copy.hyperplanes_ = new_hyperplanes
        samples_constraint_satisfied = self._check_constraint_satisfaction(test_copy, X, y, stolen)

        # no need to update hyperplanes_ids_, as the new
        # hyperplanes are just refit version of the current ones
        if samples_constraint_satisfied:
            self.hyperplanes_ = new_hyperplanes

    def combinatorial_round(self, X, y, part_x, part_y, part_sw):

        winning_score = self.score_fn[0](y, self.predict(X))
        _winning_pno = None
        winning_copy = None

        for part_id in part_x.keys():
            x_of_part = part_x[part_id]
            y_of_part = part_y[part_id]
            sw_of_part = part_sw[part_id]
            for feature in range(X.shape[1]):
                feature_sort = np.argsort(x_of_part[:, feature])
                x_of_part_sf = x_of_part[feature_sort, :]
                y_of_part_sf = y_of_part[feature_sort]
                if sw_of_part is not None:
                    sw_of_part_sf = sw_of_part[feature_sort]
                for node in range(1, len(y_of_part) - 1):
                    x_split_1 = x_of_part_sf[0:node, :]
                    y_split_1 = y_of_part_sf[0:node]
                    x_split_2 = x_of_part_sf[node:, :]
                    y_split_2 = y_of_part_sf[node:]
                    sw_split_1 = sw_of_part_sf[0:node] if sw_of_part is not None else None
                    sw_split_2 = sw_of_part_sf[node:] if sw_of_part is not None else None

                    model_1 = deepcopy(self.underlying_estimator)
                    model_1.fit(x_split_1, y_split_1, sample_weight=sw_split_1)
                    model_2 = deepcopy(self.underlying_estimator)
                    model_2.fit(x_split_2, y_split_2, sample_weight=sw_split_2)

                    # make copy with new models
                    test_copy = deepcopy(self)
                    kill_index = test_copy.hyperplanes_ids_.index(part_id)
                    del test_copy.hyperplanes_[kill_index]
                    del test_copy.hyperplanes_ids_[kill_index]
                    test_copy.hyperplanes_.extend([model_1, model_2])
                    test_copy.hyperplanes_ids_.extend([uuid4().hex for _ in range(2)])

                    constraint_satisfied = self._check_constraint_satisfaction(test_copy, X, y, {})
                    if not constraint_satisfied:
                        continue

                    # score the copy
                    y_pred = test_copy.predict(X)
                    score_of_split = self.score_fn[0](y, y_pred)

                    # the score to beat is the one of the current model.
                    if self._compare_score(score_of_split, winning_score):
                        winning_score = score_of_split
                        _winning_pno = part_id
                        winning_copy = test_copy

            return winning_copy

    def lintree_round(self, X, y, part_x, part_y, part_sw, stolen):

        # generate candidate splits, record best
        winning_score = self.score_fn[0](y, self.predict(X))
        _winning_pno = None
        winning_copy = None
        winning_data_split = None
        for pno, Xp in part_x.items():

            # check that the partition satisfies the minimum leaf samples
            assert Xp.shape[0] >= min(self.min_leaf_samples, X.shape[0]) or pno in stolen.keys(), \
                "Min leaf samples violated. This might suggest a problem with the dependency linear-tree."

            # not enough samples in this partition
            if Xp.shape[0] < 2 * self.min_leaf_samples:
                continue

            # train a LinearTreeRegressor with depth 1 (a splitter)
            # on this partition
            yp = part_y[pno]
            splitter = LinearTreeRegressor(
                base_estimator=self.underlying_estimator,
                max_depth=1,
                min_samples_leaf=self.min_leaf_samples,
            )
            psw = _check_sample_weight(part_sw[pno], part_x[pno])
            psw += 1e-12
            splitter.fit(Xp, yp, sample_weight=psw)

            # splitter did not divide the partition
            if len(_leaves(splitter)) < 2:
                continue

            assert all([x['samples'] >= self.min_leaf_samples for x in splitter.summary().values()])
            assert len(_leaves(splitter)) == 2, "Problem with tree depth"

            # make a test copy with the split implemented
            test_copy = deepcopy(self)
            kill_index = test_copy.hyperplanes_ids_.index(pno)
            del test_copy.hyperplanes_[kill_index]
            del test_copy.hyperplanes_ids_[kill_index]
            split_models, data_split = _dump_models(splitter, Xp, part_y[pno], part_sw[pno])
            for model_name, model in split_models.items():
                test_copy.hyperplanes_.append(model)
                test_copy.hyperplanes_ids_.append(model_name)

            # verify that this candidate does not violate the leaf constraint
            # this can happen if the new hyperplanes "steal" points from other leaves
            # constraint_satisfied = self._check_constraint_satisfaction(test_copy, X, y, stolen)
            # if not constraint_satisfied:
            #     continue

            # todo the Linear Tree might make a split that violates the constraint, but maybe
            # there exists a slightly less optimal split that satisfies it. This split is overseen!
            # in the paper, all splits are considered; here we use LTR. Ideally, you should constrain
            # LTR not to violate the constraint...

            # score the copy
            y_pred = test_copy.predict(X)
            score_of_split = self.score_fn[0](y, y_pred)

            # the score to beat is the one of the current model.
            if self._compare_score(score_of_split, winning_score):
                winning_score = score_of_split
                _winning_pno = pno
                winning_copy = test_copy
                winning_data_split = data_split

        return winning_copy, winning_data_split

    def prefit(self, X, y, sample_weight):
        # data validation
        X, y = self._validate_data(X, y)
        sample_weight = _check_sample_weight(sample_weight, X)

        # define/reset the fitted attributes
        self.hyperplanes_ = []  # pylint: disable=attribute-defined-outside-init
        self.hyperplanes_ids_ = []  # pylint: disable=attribute-defined-outside-init
        self.model_sequence_ = []  # pylint: disable=attribute-defined-outside-init

        # initial fit with one hyperplane
        self._initial_fit(X, y)

        return X, y, sample_weight

    def fit(self, X, y, sample_weight=None):
        raise NotImplementedError


if __name__ == '__main__':
    pass
