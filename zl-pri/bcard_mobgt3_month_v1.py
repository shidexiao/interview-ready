def bcard_mobgt3_month_v1(model_data):
  import numpy as np
  model_data['Score_last_30day_maxdueday'] = \
    np.where(model_data['last_30day_maxdueday'].isnull(),     -1.5541383774622495,
    np.where(model_data['last_30day_maxdueday'] < 0.5,        2.5240492085069093,
  -22.346092655528842))

  model_data['Score_advancerepayed_cnt'] = \
    np.where(model_data['advancerepayed_cnt'] <5.5,     -1.6217882300776592,
    np.where(model_data['advancerepayed_cnt'] < 14.0,        11.431375625475619,
  18.313125341624534))

  model_data['Score_avg_credit_lastmth1_use_rat'] = \
    np.where(model_data['avg_credit_lastmth1_use_rat'] <42.658,     21.949354068957227,
    np.where(model_data['avg_credit_lastmth1_use_rat'] < 66.8,        10.68000636246857,
    np.where(model_data['avg_credit_lastmth1_use_rat'] < 75.177,        4.802825633996915,
    np.where(model_data['avg_credit_lastmth1_use_rat'] < 80.1128,        1.397936091919867,
    np.where(model_data['avg_credit_lastmth1_use_rat'] < 95.125,        -3.191002323892962,
  -7.189526158546583)))))

  model_data['Score_closeddistance_perd2_tx_maxdueday'] = \
    np.where(model_data['closeddistance_perd2_tx_maxdueday'] <0.5,     2.0976537767701617,
  -19.1316057609262)

  model_data['Score_last60days_pass_rat'] = \
    np.where(model_data['last60days_pass_rat'].isnull(),     1.433793209529712,
    np.where(model_data['last60days_pass_rat'] < 70.835,        -10.343847839944262,
  -0.32798709876533333))

  model_data['Score_last60_advancerepayed2days_loanno_cnt'] = \
    np.where(model_data['last60_advancerepayed2days_loanno_cnt'] <0.5,     -1.26774635995695,
  14.873589005484426)

  return model_data
