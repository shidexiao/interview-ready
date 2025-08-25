def process(model_data):
  import numpy as np
  model_data['W_avg_credit_lastmth1_use_rat'] = \
    np.where(model_data['avg_credit_lastmth1_use_rat'] <42.658,     -0.490837,
    np.where(model_data['avg_credit_lastmth1_use_rat'] < 66.8,        -0.238829,
    np.where(model_data['avg_credit_lastmth1_use_rat'] < 75.177,        -0.107402,
    np.where(model_data['avg_credit_lastmth1_use_rat'] < 80.1128,        -0.031261,
    np.where(model_data['avg_credit_lastmth1_use_rat'] < 95.125,        0.071358,
  0.160774)))))

  model_data['W_avg_credit_lastmth2_use_rat'] = \
    np.where(model_data['avg_credit_lastmth2_use_rat'] <43.5502,     -0.466742,
    np.where(model_data['avg_credit_lastmth2_use_rat'] < 61.76,        -0.316637,
    np.where(model_data['avg_credit_lastmth2_use_rat'] < 82.36,        -0.098669,
    np.where(model_data['avg_credit_lastmth2_use_rat'] < 91.14,        0.043848,
  0.140666))))

  model_data['W_avg_credit_lastmth3_use_rat'] = \
    np.where(model_data['avg_credit_lastmth3_use_rat'] <46.66,     -0.441356,
    np.where(model_data['avg_credit_lastmth3_use_rat'] < 67.475,        -0.30524,
    np.where(model_data['avg_credit_lastmth3_use_rat'] < 83.0469,        -0.093477,
    np.where(model_data['avg_credit_lastmth3_use_rat'] < 83.845,        -0.007443,
    np.where(model_data['avg_credit_lastmth3_use_rat'] < 93.35,        0.062336,
  0.16967)))))

  model_data['W_avg_credit_lastmth4_use_rat'] = \
    np.where(model_data['avg_credit_lastmth4_use_rat'] <50.1969,     -0.430087,
    np.where(model_data['avg_credit_lastmth4_use_rat'] < 71.135,        -0.257514,
    np.where(model_data['avg_credit_lastmth4_use_rat'] < 84.885,        -0.067541,
    np.where(model_data['avg_credit_lastmth4_use_rat'] < 93.0518,        0.054356,
  0.152538))))

  model_data['W_avg_credit_lastmth5_use_rat'] = \
    np.where(model_data['avg_credit_lastmth5_use_rat'] <50.3385,     -0.409886,
    np.where(model_data['avg_credit_lastmth5_use_rat'] < 68.265,        -0.268456,
    np.where(model_data['avg_credit_lastmth5_use_rat'] < 83.89,        -0.089748,
    np.where(model_data['avg_credit_lastmth5_use_rat'] < 87.895,        -0.021659,
  0.099937))))

  model_data['W_min_credit_lastmth1_use_rat'] = \
    np.where(model_data['min_credit_lastmth1_use_rat'] <42.658,     -0.490837,
    np.where(model_data['min_credit_lastmth1_use_rat'] < 66.8,        -0.238829,
    np.where(model_data['min_credit_lastmth1_use_rat'] < 75.177,        -0.107402,
    np.where(model_data['min_credit_lastmth1_use_rat'] < 80.1128,        -0.031261,
    np.where(model_data['min_credit_lastmth1_use_rat'] < 95.125,        0.071358,
  0.160774)))))

  model_data['W_min_credit_lastmth2_use_rat'] = \
    np.where(model_data['min_credit_lastmth2_use_rat'] <38.766,     -0.458897,
    np.where(model_data['min_credit_lastmth2_use_rat'] < 58.19,        -0.314306,
    np.where(model_data['min_credit_lastmth2_use_rat'] < 76.695,        -0.08127,
    np.where(model_data['min_credit_lastmth2_use_rat'] < 84.72,        0.045408,
    np.where(model_data['min_credit_lastmth2_use_rat'] < 90.095,        0.101954,
  0.167163)))))

  model_data['W_min_credit_lastmth3_use_rat'] = \
    np.where(model_data['min_credit_lastmth3_use_rat'] <24.145,     -0.460891,
    np.where(model_data['min_credit_lastmth3_use_rat'] < 58.19,        -0.274802,
    np.where(model_data['min_credit_lastmth3_use_rat'] < 79.73,        -0.043044,
    np.where(model_data['min_credit_lastmth3_use_rat'] < 84.7,        0.070466,
  0.134999))))

  model_data['W_max_credit_lastmth1_use_rat'] = \
    np.where(model_data['max_credit_lastmth1_use_rat'] <42.658,     -0.490837,
    np.where(model_data['max_credit_lastmth1_use_rat'] < 66.8,        -0.238829,
    np.where(model_data['max_credit_lastmth1_use_rat'] < 75.177,        -0.107402,
    np.where(model_data['max_credit_lastmth1_use_rat'] < 80.1128,        -0.031261,
    np.where(model_data['max_credit_lastmth1_use_rat'] < 95.125,        0.071358,
  0.160774)))))

  model_data['W_credit_used_rat'] = \
    np.where(model_data['credit_used_rat'].isnull(),     -0.570715,
    np.where(model_data['credit_used_rat'] < 42.9276,        -0.450444,
    np.where(model_data['credit_used_rat'] < 58.785,        -0.225665,
    np.where(model_data['credit_used_rat'] < 67.63,        -0.138585,
    np.where(model_data['credit_used_rat'] < 75.735,        -0.054111,
  0.110839)))))

  model_data['W_his_maxdueday'] = \
    np.where(model_data['his_maxdueday'] <0.5,     -0.092816,
  0.443304)

  model_data['W_last_30day_maxdueday'] = \
    np.where(model_data['last_30day_maxdueday'].isnull(),     0.053416,
    np.where(model_data['last_30day_maxdueday'] < 0.5,        -0.086752,
  0.768039))

  model_data['W_last_60day_maxdueday'] = \
    np.where(model_data['last_60day_maxdueday'] <0.5,     -0.095057,
  0.60107)

  model_data['W_last_90day_maxdueday'] = \
    np.where(model_data['last_90day_maxdueday'] <0.5,     -0.095267,
  0.516778)

  model_data['W_last_180day_maxdueday'] = \
    np.where(model_data['last_180day_maxdueday'] <0.5,     -0.092545,
  0.445317)

  model_data['W_last30day_due1_cnt'] = \
    np.where(model_data['last30day_due1_cnt'] <0.5,     -0.083322,
  0.768039)

  model_data['W_last60day_due1_cnt'] = \
    np.where(model_data['last60day_due1_cnt'] <0.5,     -0.095057,
  0.60107)

  model_data['W_last90day_due1_cnt'] = \
    np.where(model_data['last90day_due1_cnt'] <0.5,     -0.095267,
    np.where(model_data['last90day_due1_cnt'] < 1.5,        0.329935,
  0.752637))

  model_data['W_last120day_due1_cnt'] = \
    np.where(model_data['last120day_due1_cnt'] <0.5,     -0.095209,
    np.where(model_data['last120day_due1_cnt'] < 1.5,        0.295833,
  0.69976))

  model_data['W_last150day_due1_cnt'] = \
    np.where(model_data['last150day_due1_cnt'] <0.5,     -0.093818,
    np.where(model_data['last150day_due1_cnt'] < 1.5,        0.282131,
  0.654141))

  model_data['W_last180day_due1_cnt'] = \
    np.where(model_data['last180day_due1_cnt'] <0.5,     -0.092545,
    np.where(model_data['last180day_due1_cnt'] < 1.5,        0.267758,
  0.637916))

  model_data['W_due1_cnt'] = \
    np.where(model_data['due1_cnt'] <0.5,     -0.092816,
    np.where(model_data['due1_cnt'] < 1.5,        0.267433,
  0.633817))

  model_data['W_last30day_due1_loan_cnt'] = \
    np.where(model_data['last30day_due1_loan_cnt'] <0.5,     -0.083322,
  0.768039)

  model_data['W_last60day_due1_loan_cnt'] = \
    np.where(model_data['last60day_due1_loan_cnt'] <0.5,     -0.095057,
  0.60107)

  model_data['W_last90day_due1_loan_cnt'] = \
    np.where(model_data['last90day_due1_loan_cnt'] <0.5,     -0.095267,
  0.516778)

  model_data['W_last120day_due1_loan_cnt'] = \
    np.where(model_data['last120day_due1_loan_cnt'] <0.5,     -0.095209,
  0.482757)

  model_data['W_last150day_due1_loan_cnt'] = \
    np.where(model_data['last150day_due1_loan_cnt'] <0.5,     -0.093818,
  0.458597)

  model_data['W_last180day_due1_loan_cnt'] = \
    np.where(model_data['last180day_due1_loan_cnt'] <0.5,     -0.092545,
  0.445317)

  model_data['W_due1_loan_cnt'] = \
    np.where(model_data['due1_loan_cnt'] <0.5,     -0.092816,
  0.443304)

  model_data['W_due0_loanno_rat'] = \
    np.where(model_data['due0_loanno_rat'] <36.93,     -0.080465,
  0.537181)

  model_data['W_last30_due0_loanno_rat'] = \
    np.where(model_data['last30_due0_loanno_rat'].isnull(),     0.053416,
    np.where(model_data['last30_due0_loanno_rat'] < 15.48,        -0.087523,
  0.784554))

  model_data['W_last60_due0_loanno_rat'] = \
    np.where(model_data['last60_due0_loanno_rat'] <15.48,     -0.095702,
  0.611176)

  model_data['W_last90_due0_loanno_rat'] = \
    np.where(model_data['last90_due0_loanno_rat'] <15.48,     -0.095988,
    np.where(model_data['last90_due0_loanno_rat'] < 58.335,        0.428194,
  0.630603))

  model_data['W_last120_due0_loanno_rat'] = \
    np.where(model_data['last120_due0_loanno_rat'] <15.48,     -0.096023,
    np.where(model_data['last120_due0_loanno_rat'] < 55.0,        0.400411,
  0.591266))

  model_data['W_last150_due0_loanno_rat'] = \
    np.where(model_data['last150_due0_loanno_rat'] <31.665,     -0.08884,
    np.where(model_data['last150_due0_loanno_rat'] < 55.0,        0.409741,
  0.572033))

  model_data['W_last180_due0_loanno_rat'] = \
    np.where(model_data['last180_due0_loanno_rat'] <36.93,     -0.079274,
  0.533773)

  model_data['W_last30_due0_perdno_rat'] = \
    np.where(model_data['last30_due0_perdno_rat'].isnull(),     0.053416,
    np.where(model_data['last30_due0_perdno_rat'] < 15.48,        -0.087594,
  0.785575))

  model_data['W_last60_due0_perdno_rat'] = \
    np.where(model_data['last60_due0_perdno_rat'] <8.01,     -0.095511,
  0.609413)

  model_data['W_last90_due0_perdno_rat'] = \
    np.where(model_data['last90_due0_perdno_rat'] <12.77,     -0.089751,
  0.58029)

  model_data['W_last120_due0_perdno_rat'] = \
    np.where(model_data['last120_due0_perdno_rat'] <10.265,     -0.091168,
    np.where(model_data['last120_due0_perdno_rat'] < 26.135,        0.450852,
  0.624592))

  model_data['W_last150_due0_perdno_rat'] = \
    np.where(model_data['last150_due0_perdno_rat'] <10.435,     -0.088428,
    np.where(model_data['last150_due0_perdno_rat'] < 22.875,        0.408859,
  0.601919))

  model_data['W_last180_due0_perdno_rat'] = \
    np.where(model_data['last180_due0_perdno_rat'] <10.96,     -0.085776,
    np.where(model_data['last180_due0_perdno_rat'] < 23.61,        0.417521,
  0.588881))

  model_data['W_last60_due0_up500_loanno_rat'] = \
    np.where(model_data['last60_due0_up500_loanno_rat'] <8.335,     -0.057597,
  0.665278)

  model_data['W_last90_due0_up500_loanno_rat'] = \
    np.where(model_data['last90_due0_up500_loanno_rat'] <12.5,     -0.058669,
  0.574894)

  model_data['W_last120_due0_up500_loanno_rat'] = \
    np.where(model_data['last120_due0_up500_loanno_rat'] <12.5,     -0.05759,
  0.524537)

  model_data['W_last150_due0_up500_loanno_rat'] = \
    np.where(model_data['last150_due0_up500_loanno_rat'] <12.5,     -0.055551,
  0.486971)

  model_data['W_last180_due0_up500_loanno_rat'] = \
    np.where(model_data['last180_due0_up500_loanno_rat'] <12.5,     -0.054306,
  0.465715)

  model_data['W_last60_due0_up500_perdno_rat'] = \
    np.where(model_data['last60_due0_up500_perdno_rat'] <4.545,     -0.057597,
  0.665278)

  model_data['W_last90_due0_up500_perdno_rat'] = \
    np.where(model_data['last90_due0_up500_perdno_rat'] <4.515,     -0.058669,
  0.574894)

  model_data['W_last120_due0_up500_perdno_rat'] = \
    np.where(model_data['last120_due0_up500_perdno_rat'] <4.88,     -0.057613,
  0.524824)

  model_data['W_last150_due0_up500_perdno_rat'] = \
    np.where(model_data['last150_due0_up500_perdno_rat'] <4.26,     -0.055574,
  0.487243)

  model_data['W_last180_due0_up500_perdno_rat'] = \
    np.where(model_data['last180_due0_up500_perdno_rat'] <3.925,     -0.054329,
  0.465978)

  model_data['W_closeddistance_perd2_tx_maxdueday'] = \
    np.where(model_data['closeddistance_perd2_tx_maxdueday'] <0.5,     -0.083127,
  0.758158)

  model_data['W_closeddistance_perd3_tx_maxdueday'] = \
    np.where(model_data['closeddistance_perd3_tx_maxdueday'] <0.5,     -0.092796,
  0.66844)

  model_data['W_closeddistance_perd4_tx_maxdueday'] = \
    np.where(model_data['closeddistance_perd4_tx_maxdueday'] <0.5,     -0.095608,
  0.603998)

  model_data['W_closeddistance_perd5_tx_maxdueday'] = \
    np.where(model_data['closeddistance_perd5_tx_maxdueday'] <0.5,     -0.095894,
  0.557109)

  model_data['W_closeddistance_perd6_tx_maxdueday'] = \
    np.where(model_data['closeddistance_perd6_tx_maxdueday'] <0.5,     -0.093534,
  0.513316)

  model_data['W_closeddistance_perdmth1_tx_maxdueday'] = \
    np.where(model_data['closeddistance_perdmth1_tx_maxdueday'] <0.5,     -0.092816,
  0.443304)

  model_data['W_closeddistance_perdmth2_tx_maxdueday'] = \
    np.where(model_data['closeddistance_perdmth2_tx_maxdueday'] <0.5,     -0.092816,
  0.443304)

  model_data['W_closeddistance_perdmth3_tx_maxdueday'] = \
    np.where(model_data['closeddistance_perdmth3_tx_maxdueday'] <0.5,     -0.092816,
  0.443304)

  model_data['W_closeddistance_perdmth4_tx_maxdueday'] = \
    np.where(model_data['closeddistance_perdmth4_tx_maxdueday'] <0.5,     -0.092816,
  0.443304)

  model_data['W_closeddistance_perdmth5_tx_maxdueday'] = \
    np.where(model_data['closeddistance_perdmth5_tx_maxdueday'] <0.5,     -0.092816,
  0.443304)

  model_data['W_closeddistance_perdmth6_tx_maxdueday'] = \
    np.where(model_data['closeddistance_perdmth6_tx_maxdueday'] <0.5,     -0.092816,
  0.443304)

  model_data['W_closeddistance_perdmth2_tx_ratdue1_perdno'] = \
    np.where(model_data['closeddistance_perdmth2_tx_ratdue1_perdno'] <9.41,     -0.089531,
    np.where(model_data['closeddistance_perdmth2_tx_ratdue1_perdno'] < 20.715,        0.385011,
  0.594597))

  model_data['W_closeddistance_perdmth3_tx_ratdue1_perdno'] = \
    np.where(model_data['closeddistance_perdmth3_tx_ratdue1_perdno'] <9.41,     -0.089531,
    np.where(model_data['closeddistance_perdmth3_tx_ratdue1_perdno'] < 20.715,        0.385011,
  0.594597))

  model_data['W_closeddistance_perdmth4_tx_ratdue1_perdno'] = \
    np.where(model_data['closeddistance_perdmth4_tx_ratdue1_perdno'] <9.41,     -0.089531,
    np.where(model_data['closeddistance_perdmth4_tx_ratdue1_perdno'] < 20.715,        0.385011,
  0.594597))

  model_data['W_closeddistance_perdmth5_tx_ratdue1_perdno'] = \
    np.where(model_data['closeddistance_perdmth5_tx_ratdue1_perdno'] <9.41,     -0.089531,
    np.where(model_data['closeddistance_perdmth5_tx_ratdue1_perdno'] < 20.715,        0.385011,
  0.594597))

  model_data['W_closeddistance_perdmth6_tx_ratdue1_perdno'] = \
    np.where(model_data['closeddistance_perdmth6_tx_ratdue1_perdno'] <9.41,     -0.089531,
    np.where(model_data['closeddistance_perdmth6_tx_ratdue1_perdno'] < 20.715,        0.385011,
  0.594597))

  model_data['W_closeddistance_perdmth2_tx_ratdue1_history_loan_no'] = \
    np.where(model_data['closeddistance_perdmth2_tx_ratdue1_history_loan_no'] <36.93,     -0.080465,
  0.537181)

  model_data['W_closeddistance_perdmth3_tx_ratdue1_history_loan_no'] = \
    np.where(model_data['closeddistance_perdmth3_tx_ratdue1_history_loan_no'] <36.93,     -0.080465,
  0.537181)

  model_data['W_closeddistance_perdmth4_tx_ratdue1_history_loan_no'] = \
    np.where(model_data['closeddistance_perdmth4_tx_ratdue1_history_loan_no'] <36.93,     -0.080465,
  0.537181)

  model_data['W_closeddistance_perdmth5_tx_ratdue1_history_loan_no'] = \
    np.where(model_data['closeddistance_perdmth5_tx_ratdue1_history_loan_no'] <36.93,     -0.080465,
  0.537181)

  model_data['W_closeddistance_perdmth6_tx_ratdue1_history_loan_no'] = \
    np.where(model_data['closeddistance_perdmth6_tx_ratdue1_history_loan_no'] <36.93,     -0.080465,
  0.537181)

  model_data['W_closeddistance_perdmth2_tx_ratdue1_up500_perdno'] = \
    np.where(model_data['closeddistance_perdmth2_tx_ratdue1_up500_perdno'] <3.775,     -0.05484,
  0.465855)

  model_data['W_closeddistance_perdmth3_tx_ratdue1_up500_perdno'] = \
    np.where(model_data['closeddistance_perdmth3_tx_ratdue1_up500_perdno'] <3.775,     -0.05484,
  0.465855)

  model_data['W_closeddistance_perdmth4_tx_ratdue1_up500_perdno'] = \
    np.where(model_data['closeddistance_perdmth4_tx_ratdue1_up500_perdno'] <3.775,     -0.05484,
  0.465855)

  model_data['W_closeddistance_perdmth5_tx_ratdue1_up500_perdno'] = \
    np.where(model_data['closeddistance_perdmth5_tx_ratdue1_up500_perdno'] <3.775,     -0.05484,
  0.465855)

  model_data['W_closeddistance_perdmth6_tx_ratdue1_up500_perdno'] = \
    np.where(model_data['closeddistance_perdmth6_tx_ratdue1_up500_perdno'] <3.775,     -0.05484,
  0.465855)

  model_data['W_advancerepayed_cnt'] = \
    np.where(model_data['advancerepayed_cnt'] <5.5,     0.052875,
    np.where(model_data['advancerepayed_cnt'] < 14.0,        -0.372696,
  -0.597061))

  model_data['W_last60_advancerepayed_cnt'] = \
    np.where(model_data['last60_advancerepayed_cnt'] <5.5,     0.03898,
  -0.609267)

  model_data['W_last90_advancerepayed_cnt'] = \
    np.where(model_data['last90_advancerepayed_cnt'] <5.5,     0.042997,
  -0.498909)

  model_data['W_last120_advancerepayed_cnt'] = \
    np.where(model_data['last120_advancerepayed_cnt'] <5.5,     0.048889,
  -0.50103)

  model_data['W_last150_advancerepayed_cnt'] = \
    np.where(model_data['last150_advancerepayed_cnt'] <5.5,     0.052414,
  -0.498756)

  model_data['W_last180_advancerepayed_cnt'] = \
    np.where(model_data['last180_advancerepayed_cnt'] <5.5,     0.053184,
    np.where(model_data['last180_advancerepayed_cnt'] < 14.0,        -0.405057,
  -0.59272))

  model_data['W_advancerepayed2days_cnt'] = \
    np.where(model_data['advancerepayed2days_cnt'] <1.5,     0.05629,
    np.where(model_data['advancerepayed2days_cnt'] < 14.5,        -0.354548,
  -0.595851))

  model_data['W_last60_advancerepayed2days_cnt'] = \
    np.where(model_data['last60_advancerepayed2days_cnt'] <5.5,     0.03894,
  -0.610857)

  model_data['W_last90_advancerepayed2days_cnt'] = \
    np.where(model_data['last90_advancerepayed2days_cnt'] <5.5,     0.042907,
  -0.499388)

  model_data['W_last120_advancerepayed2days_cnt'] = \
    np.where(model_data['last120_advancerepayed2days_cnt'] <1.5,     0.052326,
  -0.474434)

  model_data['W_last150_advancerepayed2days_cnt'] = \
    np.where(model_data['last150_advancerepayed2days_cnt'] <1.5,     0.055647,
  -0.467576)

  model_data['W_last180_advancerepayed2days_cnt'] = \
    np.where(model_data['last180_advancerepayed2days_cnt'] <1.5,     0.056864,
    np.where(model_data['last180_advancerepayed2days_cnt'] < 14.5,        -0.384238,
  -0.591488))

  model_data['W_advancerepayed2days_rat'] = \
    np.where(model_data['advancerepayed2days_rat'] <8.71,     0.055673,
    np.where(model_data['advancerepayed2days_rat'] < 51.19,        -0.375536,
  -0.545685))

  model_data['W_last30_advancerepayed2days_rat'] = \
    np.where(model_data['last30_advancerepayed2days_rat'].isnull(),     -0.434558,
    np.where(model_data['last30_advancerepayed2days_rat'] < 51.315,        0.039274,
  -0.625616))

  model_data['W_last60_advancerepayed2days_rat'] = \
    np.where(model_data['last60_advancerepayed2days_rat'].isnull(),     -0.362139,
    np.where(model_data['last60_advancerepayed2days_rat'] < 44.22,        0.046574,
  -0.544851))

  model_data['W_last90_advancerepayed2days_rat'] = \
    np.where(model_data['last90_advancerepayed2days_rat'].isnull(),     -0.629371,
    np.where(model_data['last90_advancerepayed2days_rat'] < 25.405,        0.049644,
  -0.443883))

  model_data['W_last120_advancerepayed2days_rat'] = \
    np.where(model_data['last120_advancerepayed2days_rat'].isnull(),     -0.482422,
    np.where(model_data['last120_advancerepayed2days_rat'] < 33.81,        0.052336,
    np.where(model_data['last120_advancerepayed2days_rat'] < 75.96,        -0.426834,
  -0.504781)))

  model_data['W_last150_advancerepayed2days_rat'] = \
    np.where(model_data['last150_advancerepayed2days_rat'].isnull(),     -0.125913,
    np.where(model_data['last150_advancerepayed2days_rat'] < 17.16,        0.057245,
    np.where(model_data['last150_advancerepayed2days_rat'] < 71.3433,        -0.366778,
  -0.498786)))

  model_data['W_last180_advancerepayed2days_rat'] = \
    np.where(model_data['last180_advancerepayed2days_rat'].isnull(),     0.309865,
    np.where(model_data['last180_advancerepayed2days_rat'] < 33.81,        0.05489,
    np.where(model_data['last180_advancerepayed2days_rat'] < 71.01,        -0.392856,
  -0.497377)))

  model_data['W_advancerepayed2days_loanno_cnt'] = \
    np.where(model_data['advancerepayed2days_loanno_cnt'] <0.5,     0.055537,
    np.where(model_data['advancerepayed2days_loanno_cnt'] < 1.5,        -0.296267,
  -0.564331))

  model_data['W_last60_advancerepayed2days_loanno_cnt'] = \
    np.where(model_data['last60_advancerepayed2days_loanno_cnt'] <0.5,     0.042855,
  -0.502788)

  model_data['W_last120_advancerepayed2days_loanno_cnt'] = \
    np.where(model_data['last120_advancerepayed2days_loanno_cnt'] <0.5,     0.05292,
  -0.428451)

  model_data['W_last150_advancerepayed2days_loanno_cnt'] = \
    np.where(model_data['last150_advancerepayed2days_loanno_cnt'] <0.5,     0.056873,
    np.where(model_data['last150_advancerepayed2days_loanno_cnt'] < 1.5,        -0.363994,
  -0.551293))

  model_data['W_last180_advancerepayed2days_loanno_cnt'] = \
    np.where(model_data['last180_advancerepayed2days_loanno_cnt'] <0.5,     0.056791,
    np.where(model_data['last180_advancerepayed2days_loanno_cnt'] < 1.5,        -0.326691,
  -0.5661))

  model_data['W_advancerepayed2days_loanno_rat'] = \
    np.where(model_data['advancerepayed2days_loanno_rat'] <0.985,     0.055757,
  -0.392784)

  model_data['W_last30_advancerepayed2days_loanno_rat'] = \
    np.where(model_data['last30_advancerepayed2days_loanno_rat'].isnull(),     -0.434558,
    np.where(model_data['last30_advancerepayed2days_loanno_rat'] < 10.555,        0.039301,
  -0.555855))

  model_data['W_last60_advancerepayed2days_loanno_rat'] = \
    np.where(model_data['last60_advancerepayed2days_loanno_rat'].isnull(),     -0.362139,
    np.where(model_data['last60_advancerepayed2days_loanno_rat'] < 10.555,        0.046928,
  -0.505663))

  model_data['W_last90_advancerepayed2days_loanno_rat'] = \
    np.where(model_data['last90_advancerepayed2days_loanno_rat'].isnull(),     -0.629371,
    np.where(model_data['last90_advancerepayed2days_loanno_rat'] < 11.805,        0.049227,
  -0.420722))

  model_data['W_last120_advancerepayed2days_loanno_rat'] = \
    np.where(model_data['last120_advancerepayed2days_loanno_rat'].isnull(),     -0.482422,
    np.where(model_data['last120_advancerepayed2days_loanno_rat'] < 10.555,        0.054039,
  -0.429932))

  model_data['W_last150_advancerepayed2days_loanno_rat'] = \
    np.where(model_data['last150_advancerepayed2days_loanno_rat'].isnull(),     -0.125913,
    np.where(model_data['last150_advancerepayed2days_loanno_rat'] < 11.805,        0.057292,
  -0.43362))

  model_data['W_last180_advancerepayed2days_loanno_rat'] = \
    np.where(model_data['last180_advancerepayed2days_loanno_rat'].isnull(),     0.309865,
    np.where(model_data['last180_advancerepayed2days_loanno_rat'] < 11.805,        0.057048,
  -0.415493))

  model_data['W_advancerepayed5days_cnt'] = \
    np.where(model_data['advancerepayed5days_cnt'] <1.5,     0.05396,
    np.where(model_data['advancerepayed5days_cnt'] < 14.5,        -0.359604,
  -0.606973))

  model_data['W_last60_advancerepayed5days_cnt'] = \
    np.where(model_data['last60_advancerepayed5days_cnt'] <1.5,     0.040272,
  -0.598108)

  model_data['W_last90_advancerepayed5days_cnt'] = \
    np.where(model_data['last90_advancerepayed5days_cnt'] <3.5,     0.043031,
  -0.49668)

  model_data['W_last120_advancerepayed5days_cnt'] = \
    np.where(model_data['last120_advancerepayed5days_cnt'] <3.5,     0.049491,
  -0.504595)

  model_data['W_last150_advancerepayed5days_cnt'] = \
    np.where(model_data['last150_advancerepayed5days_cnt'] <0.5,     0.055838,
  -0.472501)

  model_data['W_last180_advancerepayed5days_cnt'] = \
    np.where(model_data['last180_advancerepayed5days_cnt'] <3.5,     0.053134,
  -0.479107)

  model_data['W_advancerepayed5days_rat'] = \
    np.where(model_data['advancerepayed5days_rat'] <6.04,     0.05559,
    np.where(model_data['advancerepayed5days_rat'] < 46.875,        -0.371978,
  -0.519626))

  model_data['W_last30_advancerepayed5days_rat'] = \
    np.where(model_data['last30_advancerepayed5days_rat'].isnull(),     -0.434558,
    np.where(model_data['last30_advancerepayed5days_rat'] < 42.465,        0.038702,
  -0.660218))

  model_data['W_last60_advancerepayed5days_rat'] = \
    np.where(model_data['last60_advancerepayed5days_rat'].isnull(),     -0.362139,
    np.where(model_data['last60_advancerepayed5days_rat'] < 28.87,        0.045489,
  -0.578163))

  model_data['W_last90_advancerepayed5days_rat'] = \
    np.where(model_data['last90_advancerepayed5days_rat'].isnull(),     -0.629371,
    np.where(model_data['last90_advancerepayed5days_rat'] < 38.28,        0.046691,
  -0.489144))

  model_data['W_last120_advancerepayed5days_rat'] = \
    np.where(model_data['last120_advancerepayed5days_rat'].isnull(),     -0.482422,
    np.where(model_data['last120_advancerepayed5days_rat'] < 20.225,        0.052808,
    np.where(model_data['last120_advancerepayed5days_rat'] < 78.87,        -0.448534,
  -0.52386)))

  model_data['W_last150_advancerepayed5days_rat'] = \
    np.where(model_data['last150_advancerepayed5days_rat'].isnull(),     -0.125913,
    np.where(model_data['last150_advancerepayed5days_rat'] < 17.16,        0.055801,
    np.where(model_data['last150_advancerepayed5days_rat'] < 70.71,        -0.426104,
  -0.519383)))

  model_data['W_last180_advancerepayed5days_rat'] = \
    np.where(model_data['last180_advancerepayed5days_rat'].isnull(),     0.309865,
    np.where(model_data['last180_advancerepayed5days_rat'] < 16.955,        0.055642,
    np.where(model_data['last180_advancerepayed5days_rat'] < 71.01,        -0.405692,
  -0.514473)))

  model_data['W_advancerepayed5days_loanno_cnt'] = \
    np.where(model_data['advancerepayed5days_loanno_cnt'] <0.5,     0.055183,
    np.where(model_data['advancerepayed5days_loanno_cnt'] < 1.5,        -0.343249,
  -0.586448))

  model_data['W_last60_advancerepayed5days_loanno_cnt'] = \
    np.where(model_data['last60_advancerepayed5days_loanno_cnt'] <0.5,     0.041549,
  -0.563551)

  model_data['W_last90_advancerepayed5days_loanno_cnt'] = \
    np.where(model_data['last90_advancerepayed5days_loanno_cnt'] <0.5,     0.045268,
  -0.462368)

  model_data['W_last120_advancerepayed5days_loanno_cnt'] = \
    np.where(model_data['last120_advancerepayed5days_loanno_cnt'] <0.5,     0.052031,
  -0.472355)

  model_data['W_last150_advancerepayed5days_loanno_cnt'] = \
    np.where(model_data['last150_advancerepayed5days_loanno_cnt'] <0.5,     0.055838,
  -0.472501)

  model_data['W_last180_advancerepayed5days_loanno_cnt'] = \
    np.where(model_data['last180_advancerepayed5days_loanno_cnt'] <0.5,     0.05609,
    np.where(model_data['last180_advancerepayed5days_loanno_cnt'] < 1.5,        -0.377506,
  -0.58249))

  model_data['W_advancerepayed5days_loanno_rat'] = \
    np.where(model_data['advancerepayed5days_loanno_rat'] <0.985,     0.055401,
    np.where(model_data['advancerepayed5days_loanno_rat'] < 4.585,        -0.390665,
  -0.484878))

  model_data['W_last30_advancerepayed5days_loanno_rat'] = \
    np.where(model_data['last30_advancerepayed5days_loanno_rat'].isnull(),     -0.434558,
    np.where(model_data['last30_advancerepayed5days_loanno_rat'] < 10.555,        0.038689,
  -0.645029))

  model_data['W_last60_advancerepayed5days_loanno_rat'] = \
    np.where(model_data['last60_advancerepayed5days_loanno_rat'].isnull(),     -0.362139,
    np.where(model_data['last60_advancerepayed5days_loanno_rat'] < 10.555,        0.045566,
  -0.566967))

  model_data['W_last90_advancerepayed5days_loanno_rat'] = \
    np.where(model_data['last90_advancerepayed5days_loanno_rat'].isnull(),     -0.629371,
    np.where(model_data['last90_advancerepayed5days_loanno_rat'] < 11.805,        0.048089,
  -0.465372))

  model_data['W_last120_advancerepayed5days_loanno_rat'] = \
    np.where(model_data['last120_advancerepayed5days_loanno_rat'].isnull(),     -0.482422,
    np.where(model_data['last120_advancerepayed5days_loanno_rat'] < 10.555,        0.053135,
    np.where(model_data['last120_advancerepayed5days_loanno_rat'] < 77.5,        -0.439324,
  -0.524006)))

  model_data['W_last150_advancerepayed5days_loanno_rat'] = \
    np.where(model_data['last150_advancerepayed5days_loanno_rat'].isnull(),     -0.125913,
    np.where(model_data['last150_advancerepayed5days_loanno_rat'] < 11.805,        0.056252,
  -0.476686))

  model_data['W_last180_advancerepayed5days_loanno_rat'] = \
    np.where(model_data['last180_advancerepayed5days_loanno_rat'].isnull(),     0.309865,
    np.where(model_data['last180_advancerepayed5days_loanno_rat'] < 11.805,        0.056344,
    np.where(model_data['last180_advancerepayed5days_loanno_rat'] < 55.0,        -0.417183,
  -0.500558)))

  model_data['W_settled_loanno_cnt'] = \
    np.where(model_data['settled_loanno_cnt'] <0.5,     0.052223,
    np.where(model_data['settled_loanno_cnt'] < 1.5,        -0.360814,
  -0.620656))

  model_data['W_settled_loanno_rat'] = \
    np.where(model_data['settled_loanno_rat'] <11.805,     0.05244,
    np.where(model_data['settled_loanno_rat'] < 55.0,        -0.39361,
  -0.561279))

  model_data['W_last60days_pass_rat'] = \
    np.where(model_data['last60days_pass_rat'].isnull(),     -0.074005,
    np.where(model_data['last60days_pass_rat'] < 70.835,        0.533896,
  0.016929))

  model_data['W_last90days_pass_rat'] = \
    np.where(model_data['last90days_pass_rat'].isnull(),     -0.127606,
    np.where(model_data['last90days_pass_rat'] < 77.5,        0.484377,
  -0.004171))

  model_data['W_last120days_pass_rat'] = \
    np.where(model_data['last120days_pass_rat'].isnull(),     -0.217095,
    np.where(model_data['last120days_pass_rat'] < 84.52,        0.463674,
  -0.014802))

  return model_data
