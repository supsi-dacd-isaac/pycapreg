from sklearn.utils.estimator_checks import check_estimator
from pycapreg import LTCAPRegressor, ClCAPRegressor

if __name__ == '__main__':

    cap_regressor = ClCAPRegressor()
    check_estimator(cap_regressor)

    cap_regressor = LTCAPRegressor()
    check_estimator(cap_regressor)
