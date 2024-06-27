from copy import deepcopy
from warnings import warn

from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import mean_squared_error

from .basecapregressor import _BaseCAPRegressor, FEW_SAMPLES_MSG, IT_LIM_HIT_MSG, _subsets_to_freeze


class LTCAPRegressor(_BaseCAPRegressor):

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

        super().__init__(
                concavity=concavity,
                min_leaf_samples=min_leaf_samples,
                save_model_sequence=save_model_sequence,
                score_fn=score_fn,
                refit_rounds=refit_rounds,
                emergency_iteration_limit=emergency_iteration_limit,
        )

        self.attempt_full_cap_on_stop = attempt_full_cap_on_stop

    def fit(self, X, y, sample_weight=None):
        # data validation
        X, y, sample_weight = self.prefit(X, y, sample_weight)

        # warning in case of very few samples
        if X.shape[0] < 2*self.min_leaf_samples:
            warn(
                FEW_SAMPLES_MSG.format(
                    min_leaf_samples=self.min_leaf_samples,
                    nsamples=X.shape[0],
                    nsamples2=2*self.min_leaf_samples
                ),
                FitFailedWarning)

        frozen_subsets = {}
        for _nit in self._generate_iterations_number():
            # obtain the partitioning induced by the current hyperplanes
            part_x, part_y, part_sw = self.partition_data(X, y, sample_weight)

            # perform a round of lintree split
            winning_copy, winning_data_split = self.lintree_round(X, y, part_x, part_y, part_sw, frozen_subsets)

            # if lintree split did not achieve an improvement, try a classic cap
            if winning_copy is None and self.attempt_full_cap_on_stop:
                winning_copy = self.combinatorial_round(X, y, part_x, part_y, part_sw)

            if winning_copy is None:
                # no candidate split has beaten the current model.
                # the iterations stop.
                break

            # if any hyperplanes fall under the minimum leaf requirement,
            # freeze the subsets they are currently fit on
            frozen_subsets = _subsets_to_freeze(self, winning_copy, X, y, sample_weight, self.min_leaf_samples, frozen_subsets, winning_data_split)

            # split the winning partition.
            # partition numbers have the same order as the hyperplanes they are induced by.
            # therefore you substitute the hyperplane at index winning_pno
            self.hyperplanes_ = winning_copy.hyperplanes_
            self.hyperplanes_ids_ = winning_copy.hyperplanes_ids_

            # refit
            for _ in range(self.refit_rounds):
                self.refit_current_hyperplanes(X, y, frozen_subsets, sample_weight=sample_weight)

            # purge planes with no leaves
            self.purge_dead_hyperplanes(X, y)

            if self.save_model_sequence:
                self.model_sequence_.append(deepcopy(self))

        else:
            # fitting did not end naturally
            warn(IT_LIM_HIT_MSG, FitFailedWarning)

        return self
