from copy import deepcopy

from sklearn.exceptions import FitFailedWarning

from .basecapregressor import _BaseCAPRegressor, IT_LIM_HIT_MSG, warn


class ClCAPRegressor(_BaseCAPRegressor):

    def fit(self, X, y, sample_weight=None):
        X, y, sample_weight = self._prefit(X, y, sample_weight)

        for _nit in self._generate_iterations_number():
            part_x, part_y, part_sw = self.partition_data(X, y, sample_weight)
            winning_copy = self.combinatorial_round(X, y, part_x, part_y, part_sw)
            if winning_copy is None:
                # no candidate split has beaten the current model.
                # the iterations stop.
                break

            # split the winning partition.
            # partition numbers have the same order as the hyperplanes they are induced by.
            # therefore you substitute the hyperplane at index winning_pno
            self.hyperplanes_ = winning_copy.hyperplanes_
            self.hyperplanes_ids_ = winning_copy.hyperplanes_ids_

            # refit
            for _ in range(self.refit_rounds):
                skipped_planes = self.refit_current_hyperplanes(X, y, sample_weight=sample_weight)
                assert not skipped_planes
                # planes are skipped if they ended up underweight in the previous iteration.
                # this must not happen with only combinatorial rounds.

            if self.save_model_sequence:
                self.model_sequence_.append(deepcopy(self))

        else:
            # fitting did not end naturally
            warn(IT_LIM_HIT_MSG, FitFailedWarning)

        return self


if __name__ == '__main__':
    from sklearn.utils.estimator_checks import check_estimator
    cap_regressor = ClCAPRegressor()
    check_estimator(cap_regressor)