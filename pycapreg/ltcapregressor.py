from copy import deepcopy
from warnings import warn

from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import mean_squared_error

from .basecapregressor import _BaseCAPRegressor, FEW_SAMPLES_MSG, IT_LIM_HIT_MSG


class LTCAPRegressor(_BaseCAPRegressor):

    def __init__(
            self,
            concavity='convex',
            min_leaf_samples=3,
            break_if_no_improvement=True,
            attempt_full_cap_on_stop=False,
            save_model_sequence=False,
            score_fn=(mean_squared_error, 'lower'),
            refit_rounds=1,
            emergency_iteration_limit=1000,
    ):

        super().__init__(
                concavity=concavity,
                min_leaf_samples=min_leaf_samples,
                break_if_no_improvement=break_if_no_improvement,
                save_model_sequence=save_model_sequence,
                score_fn=score_fn,
                refit_rounds=refit_rounds,
                emergency_iteration_limit=emergency_iteration_limit,
        )

        self.attempt_full_cap_on_stop = attempt_full_cap_on_stop

    def fit(self, X, y, sample_weight=None):
        # data validation
        X, y, sample_weight = self._prefit(X, y, sample_weight)

        # warning in case of very few samples
        if X.shape[0] < 2*self.get_min_samples(X):
            warn(
                FEW_SAMPLES_MSG.format(
                    min_leaf_samples=self.get_min_samples(X),
                    nsamples=X.shape[0],
                    nsamples2=2*self.get_min_samples(X)
                ),
                FitFailedWarning)

        skipped_planes = []
        for _nit in self._generate_iterations_number():
            # obtain the partitioning induced by the current hyperplanes
            part_x, part_y, part_sw = self.partition_data(X, y, sample_weight)

            # break if all partitions cannot be divided
            if all([len(y) < self.get_min_samples(X) for y in part_y.values()]):
                break

            # perform a round of lintree split
            winning_copy = self.lintree_round(X, y, part_x, part_y, part_sw, skipped_planes)

            # if lintree split did not achieve an improvement, try a classic cap
            if winning_copy is None and self.attempt_full_cap_on_stop:
                winning_copy = self.combinatorial_round(X, y, part_x, part_y, part_sw)

            if winning_copy is None:
                # no candidate split has beaten the current model.
                # the iterations stop.
                break

            # if any hyperplanes fall under the minimum leaf requirement,
            # freeze the subsets they are currently fit on
            # frozen_subsets = _subsets_to_freeze(self, winning_copy, X, y, sample_weight, self.get_min_samples(X, frozen_subsets)

            # split the winning partition.
            # partition numbers have the same order as the hyperplanes they are induced by.
            # therefore you substitute the hyperplane at index winning_pno
            self.hyperplanes_ = winning_copy.hyperplanes_
            self.hyperplanes_ids_ = winning_copy.hyperplanes_ids_

            # refit
            for _ in range(self.refit_rounds):
                skipped_planes = self.refit_current_hyperplanes(X, y, sample_weight=sample_weight)

            if self.save_model_sequence:
                self.model_sequence_.append(deepcopy(self))

        else:
            # fitting did not end naturally
            warn(IT_LIM_HIT_MSG, FitFailedWarning)

        # purge planes with no leaves
        self.purge_dead_hyperplanes(X, y)

        return self
