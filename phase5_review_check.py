import json
from pathlib import Path

import numpy as np
import pandas as pd


notebook = json.loads(Path('notebooks/06_baseline_models.ipynb').read_text(encoding='utf-8'))
scope = {'__name__': '__main__'}
for index in list(range(17)) + [20, 23, 26]:
    cell = notebook['cells'][index]
    if cell['cell_type'] == 'code':
        exec(''.join(cell['source']), scope)

validation = scope['validation_rows'].sort_values(['unit_number', 'time_in_cycles'])
actual = scope['y_test']
for name, key in [('Linear Regression', 'Lin_Reg_validation_predictions'), ('Random Forest', 'Ran_Forest_validation_predictions'), ('XGBoost', 'xgb_validation_predictions')]:
    prediction = pd.Series(scope[key], index=actual.index)
    rows = validation[['unit_number', 'time_in_cycles', 'remaining_useful_life']].copy()
    rows['predicted_rul'] = prediction
    changes = rows.groupby('unit_number')['predicted_rul'].diff().dropna()
    critical = rows['remaining_useful_life'] <= 10
    healthy = rows['remaining_useful_life'] > 30
    warning = rows['remaining_useful_life'].between(11, 30)
    alerts = rows['predicted_rul'] <= 30
    first = rows[alerts].groupby('unit_number').first()
    in_window = rows[warning & alerts]['unit_number'].nunique()
    print(name)
    print('MAE', round((actual - prediction).abs().mean(), 2))
    print('RMSE', round(np.sqrt(((actual - prediction) ** 2).mean()), 2))
    print('mean abs change', round(changes.abs().mean(), 2), 'increase pct', round(changes.gt(0).mean() * 100, 2), 'max jump', round(changes.abs().max(), 2))
    print('critical misses', int((critical & ~alerts).sum()), 'false alarm observations', int((healthy & alerts).sum()))
    print('any warning within 11-30', in_window, '/', rows['unit_number'].nunique())
    print('first alert premature/timely/late/none', int((first['remaining_useful_life'] > 30).sum()), int(first['remaining_useful_life'].between(11, 30).sum()), int((first['remaining_useful_life'] <= 10).sum()), rows['unit_number'].nunique() - len(first))
    print('first alert lead', round(first['remaining_useful_life'].mean(), 2))
