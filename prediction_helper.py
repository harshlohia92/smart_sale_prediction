import pandas as pd
import joblib

model_data = joblib.load('coding/model.joblib')
model = model_data['model']
scaler = model_data['scale']
MODEL_COLUMNS = model_data['feature']
NUMERIC_COLS = model_data['cols_to_scale']

country_map = {
    'United Kingdom': 0, 'France': 1, 'Australia': 2, 'Germany': 3, 'Norway': 4,
    'EIRE': 5, 'Switzerland': 6, 'Poland': 7, 'Portugal': 8, 'Italy': 9, 'Belgium': 10,
    'Lithuania': 11, 'Japan': 12, 'Iceland': 13, 'Channel Islands': 14, 'Denmark': 15,
    'Spain': 16, 'Cyprus': 17, 'Austria': 18, 'Sweden': 19, 'Netherlands': 20, 'Israel': 21,
    'Finland': 22, 'Greece': 23, 'Hong Kong': 24, 'Singapore': 25, 'Lebanon': 26,
    'United Arab Emirates': 27, 'Saudi Arabia': 28, 'Czech Republic': 29, 'Canada': 30,
    'Unspecified': 31, 'Brazil': 32, 'USA': 33, 'European Community': 34, 'Bahrain': 35,
    'Malta': 36, 'RSA': 37
}

week_map = {'Wednesday': 0, 'Thursday': 1, 'Friday': 2, 'Sunday': 3, 'Monday': 4, 'Tuesday': 5}
supplier_map = {'Supplier_1': 0, 'Supplier_2': 1, 'Supplier_4': 2, 'Supplier_3': 3, 'Supplier_5': 4}

MAX_SALES = 1000
PREDICTION_SCALE = 10

def prepare_input(quantity, unitprice, country, months, week, supplier,
                  supplier_avg_sale, is_christmas, is_newyear, rolling_mean_30_x, is_weekend):

    country_mapped = country_map.get(country, 31)
    week_mapped = week_map.get(week, 0)
    supplier_mapped = supplier_map.get(supplier, 0)

    input_data = {
        'quantity': quantity,
        'unitprice': unitprice,
        'country': country_mapped,
        'months': months,
        'week': week_mapped,
        'supplier': supplier_mapped,
        'supplier_avg_sale': supplier_avg_sale,
        'is_christmas': int(is_christmas),
        'is_newyear': int(is_newyear),
        'rolling_mean_30_x': rolling_mean_30_x,
        'is_weekend': int(is_weekend)
    }

    df = pd.DataFrame([input_data])


    if scaler is not None and NUMERIC_COLS is not None:
        existing_num = [c for c in NUMERIC_COLS if c in df.columns]
        if existing_num:
            df_loc = df[existing_num].astype(float)
            try:
                df.loc[:, existing_num] = scaler.transform(df_loc)
            except Exception as e:
                raise RuntimeError(f"Error applying scaler on {existing_num}: {e}")


    if MODEL_COLUMNS is not None:
        for col in MODEL_COLUMNS:
            if col not in df.columns:
                df[col] = 0
        df = df[MODEL_COLUMNS]

    return df


def predict_sales(quantity, unitprice, country, months, week, supplier,
                  supplier_avg_sale, is_christmas, is_newyear, rolling_mean_30_x, is_weekend):

    input_df = prepare_input(quantity, unitprice, country, months, week, supplier,
                             supplier_avg_sale, is_christmas, is_newyear, rolling_mean_30_x, is_weekend)

    pred = model.predict(input_df)
    prediction_scaled = float(pred[0]) if hasattr(pred, "__len__") else float(pred)

    # Rescale to actual units
    prediction_actual = prediction_scaled * MAX_SALES / PREDICTION_SCALE
    return prediction_actual