import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Set page config
st.set_page_config(layout="wide", page_title="Predicción de riesgo de Churn de afiliados")

st.title("Predicción de Riesgo de Churn de Afiliados")
st.write("Ingrese los detalles del afiliado para predecir si existe riesgo de que deje de usar los servicios ofrecidos por la caja dentro de los próximos 3 meses.")

# Load models

scaler = joblib.load('minmax_scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')
random_forest_model = joblib.load('random_forest_model.joblib')


# Streamlit input widgets
st.sidebar.header("Información del Afiliado")

EdadAfil = st.sidebar.slider('Edad Afiliado', min_value=0, max_value=80, value=30, step=1)
MunicipioAfil = st.sidebar.selectbox('Municipio', ["MEDELLÍN"])
TipoTit  = st.sidebar.selectbox('Tipo de Titular', ["Afiliado Dependiente","Afiliado Facultativo", "Afiliado Independiente","Afiliado Pensionado"])
categoriaAfil = st.sidebar.selectbox('Categoría Afiliado', ["A","B", "C"])
salario = st.sidebar.slider('Salario', min_value=0.0, max_value=100000000.0, value=1000000.0, step=1000.0)
EstadoCivilAfil =  st.sidebar.selectbox('Estado Civil Afiliado', ["Casado", "Separado", "Soltero", "Unión libre", "Viudo / a"])
GeneroAfil = st.sidebar.selectbox('Género Afiliado', ["Femenino","Masculino", "desconocido"])
Cantidad_dependientes = st.sidebar.slider('Cantidad de Dependientes', min_value=0, max_value=20, value=0, step=1)
cantidad_hijos = st.sidebar.slider('Cantidad de Hijos', min_value=0, max_value=20, value=0, step=1)
SectorEmp = st.sidebar.selectbox('Sector Empleador', ["No indica","Privada", "Pública", "Mixta"])
SeccionEconomicaEmp = st.sidebar.selectbox('Sección Económica Empleador', ["C_Industrias Manufactureras","L_Actividades inmobiliarias","M_Act. profesionales, cie. tec",
"N_Serv. Administra. y de apoyo","F_Construcción","G_Comercio y Rep. Automotores",
"I_Alojamiento, serv. de comida","P_Educación","Q_Actv de atención de la Salud",
"T_Actv Hogares en calidad empl","S_Otras Actividades de serv","H_Transporte y almacenamiento",
"J_Informacion y comunicaciones","K_Activ. financieras y seguros","E_Distr Agua y Saneamiento amb",
"A_Agric, Gan, Caza,Silvi,Pes","O_Admon Pública y Seg. Social","B_Explota. de minas y canteras",
"D_Sum. elec., gas,vapor,aire","No indica","R_Actv artisticas y Recreación","U_Organiz Extraterritoriales"])
TamanoEmp = st.sidebar.selectbox('Tamaño Empleador', ["No indica","Gran empresa", "Mediana empresa", "Pequeña empresa", "Microempresa"])
Num_servicios_6m = st.sidebar.slider('Número de Servicios en 6 meses', min_value=0, max_value=20, value=0, step=1)
total_usos_6m = st.sidebar.slider('Total de Usos en 6 meses', min_value=0, max_value=500, value=0, step=1)
total_subsidio_6m = st.sidebar.slider('Total de Subsidio en 6 meses', min_value=0.0, max_value=5000000.0, value=0.0, step=0.1)
Total_pago_6m = st.sidebar.slider('Total de Pagos en 6 meses', min_value=0.0, max_value=5000000.0, value=0.0, step=0.1)
Uso_Ultimos_3_Meses = st.sidebar.selectbox('Uso Últimos 3 Meses', ["SI","NO"])
Num_servicios_3m = st.sidebar.slider('Número de Servicios en 3 meses', min_value=0, max_value=20, value=0, step=1)
total_usos_3m = st.sidebar.slider('Total de Usos en 3 meses', min_value=0, max_value=500, value=0, step=1)
total_subsidio_3m = st.sidebar.slider('Total de Subsidio en 3 meses', min_value=0.0, max_value=5000000.0, value=0.0, step=0.1)
Total_pago_3m = st.sidebar.slider('Total de Pagos en 3 meses', min_value=0.0, max_value=5000000.0, value=0.0, step=0.1)
Num_servicios_3prev = st.sidebar.slider('Número de Servicios 3 meses Previos', min_value=0, max_value=20, value=0, step=1)
total_usos_3prev = st.sidebar.slider('Total de Usos 3 meses Previos', min_value=0, max_value=500, value=0, step=1)
total_subsidio_3prev = st.sidebar.slider('Total de Subsidio 3 meses Previos', min_value=0.0, max_value=5000000.0, value=0.0, step=0.1)
Total_pago_3prev = st.sidebar.slider('Total de Pagos 3 meses Previos', min_value=0.0, max_value=5000000.0, value=0.0, step=0.1)

if st.button('Predecir Churn'):
    # Dataframe creation
    datos = [[EdadAfil, MunicipioAfil, TipoTit, categoriaAfil, salario,
           EstadoCivilAfil, GeneroAfil, Cantidad_dependientes,
           cantidad_hijos, SectorEmp, SeccionEconomicaEmp, TamanoEmp,
           Num_servicios_6m, total_usos_6m, total_subsidio_6m,
           Total_pago_6m, Uso_Ultimos_3_Meses, Num_servicios_3m,
           total_usos_3m, total_subsidio_3m, Total_pago_3m,
           Num_servicios_3prev, total_usos_3prev, total_subsidio_3prev,
           Total_pago_3prev]]
    data = pd.DataFrame(datos, columns=['EdadAfil', 'MunicipioAfil', 'TipoTit', 'categoriaAfil', 'salario',
           'EstadoCivilAfil', 'GeneroAfil', 'Cantidad_dependientes',
           'cantidad_hijos', 'SectorEmp', 'SeccionEconomicaEmp', 'TamanoEmp',
           'Num_servicios_6m', 'total_usos_6m', 'total_subsidio_6m',
           'Total_pago_6m', 'Uso_Ultimos_3_Meses', 'Num_servicios_3m',
           'total_usos_3m', 'total_subsidio_3m', 'Total_pago_3m',
           'Num_servicios_3prev', 'total_usos_3prev', 'total_subsidio_3prev',
           'Total_pago_3prev']) 

    # Feature Engineering
    data["ratio_aporte_salario"] = np.log(data['Total_pago_6m'] + 1) - np.log(data['salario'] * 6 + 1)
    data['ratio_subsidio_salario'] = np.log(data['total_subsidio_6m'] + 1) - np.log(data['salario'] * 6 + 1)
    data['ratio_subsidio_aporte'] = np.log(data['total_subsidio_6m'] + 1) - np.log(data['Total_pago_6m'] + 1)

    data["tasa_usos"] = (data["total_usos_3m"] - data["total_usos_3prev"]) / data["total_usos_3prev"]
    data.loc[data["total_usos_3prev"] == 0, "tasa_usos"] = np.where(
        data.loc[data["total_usos_3prev"] == 0, "total_usos_3m"] == 0,
        0,
        1
    )
    data["tasa_subsidio"] = (data["total_subsidio_3m"] - data["total_subsidio_3prev"]) / data["total_subsidio_3prev"]
    data.loc[data["total_subsidio_3prev"] == 0, "tasa_subsidio"] = np.where(
        data.loc[data["total_subsidio_3prev"] == 0, "total_subsidio_3m"] == 0,
        0,
        1
    )
    data["tasa_pagos"] = (data["Total_pago_3m"] - data["Total_pago_3prev"]) / data["Total_pago_3prev"]
    data.loc[data["Total_pago_3prev"] == 0, "tasa_pagos"] = np.where(
        data.loc[data["Total_pago_3prev"] == 0, "Total_pago_3m"] == 0,
        0,
        1
    )
    data["tasa_servicios"] = (data["Num_servicios_3m"] - data["Num_servicios_3prev"]) / data["Num_servicios_3prev"]
    data.loc[data["Num_servicios_3prev"] == 0, "tasa_servicios"] = np.where(
        data.loc[data["Num_servicios_3prev"] == 0, "Num_servicios_3m"] == 0,
        0,
        1
    )

    # Select relevant features for modeling (excluding 'churn' and raw financial values that were transformed)
    selected_features_before_ohe = ['EdadAfil', 'TipoTit', 'categoriaAfil',
           'EstadoCivilAfil', 'GeneroAfil', 'Cantidad_dependientes',
           'cantidad_hijos', 'SectorEmp', 'SeccionEconomicaEmp', 'TamanoEmp',
           'Num_servicios_6m', 'total_usos_6m', 'Uso_Ultimos_3_Meses', 'ratio_aporte_salario',
           'ratio_subsidio_salario', 'ratio_subsidio_aporte', 'tasa_usos',
           'tasa_subsidio', 'tasa_pagos', 'tasa_servicios'] 

    data_preparada = data[selected_features_before_ohe].copy()

    # One-hot encoding
    data_num = pd.get_dummies(data_preparada, columns=['Uso_Ultimos_3_Meses'], drop_first=True, dtype = int)
    data_num = pd.get_dummies(data_num, columns=['TipoTit','categoriaAfil','EstadoCivilAfil','GeneroAfil','SectorEmp','SeccionEconomicaEmp',
                                                 'TamanoEmp'], drop_first=False, dtype = int)

    # Define the final columns expected by the model
    # Corrected 'Unión Libre' to 'Unión libre' for consistency with input
    columns_final = ['EdadAfil', 'Cantidad_dependientes', 'cantidad_hijos',
           'Num_servicios_6m', 'total_usos_6m', 'ratio_aporte_salario',
           'ratio_subsidio_aporte', 'tasa_usos', 'tasa_subsidio', 'tasa_pagos',
           'tasa_servicios', 'Uso_Ultimos_3_Meses_SI',
           'TipoTit_Afiliado Facultativo', 'TipoTit_Afiliado Independiente',
           'TipoTit_Afiliado Pensionado', 'categoriaAfil_A', 'categoriaAfil_B',
           'categoriaAfil_C', 'EstadoCivilAfil_Casado', 'EstadoCivilAfil_Separado',
           'EstadoCivilAfil_Soltero', 'EstadoCivilAfil_Unión libre',
           'EstadoCivilAfil_Viudo / a', 'GeneroAfil_Masculino',
           'SectorEmp_Mixta', 
           'SectorEmp_Privada', 'SectorEmp_Pública',
           'SeccionEconomicaEmp_A_Agric, Gan, Caza,Silvi,Pes',
           'SeccionEconomicaEmp_B_Explota. de minas y canteras',
           'SeccionEconomicaEmp_C_Industrias Manufactureras',
           'SeccionEconomicaEmp_D_Sum. elec., gas,vapor,aire',
           'SeccionEconomicaEmp_E_Distr Agua y Saneamiento amb',
           'SeccionEconomicaEmp_F_Construcción',
           'SeccionEconomicaEmp_G_Comercio y Rep. Automotores',
           'SeccionEconomicaEmp_H_Transporte y almacenamiento',
           'SeccionEconomicaEmp_I_Alojamiento, serv. de comida',
           'SeccionEconomicaEmp_J_Informacion y comunicaciones',
           'SeccionEconomicaEmp_K_Activ. financieras y seguros',
           'SeccionEconomicaEmp_L_Actividades inmobiliarias',
           'SeccionEconomicaEmp_M_Act. profesionales, cie. tec',
           'SeccionEconomicaEmp_N_Serv. Administra. y de apoyo',
           'SeccionEconomicaEmp_O_Admon Pública y Seg. Social',
           'SeccionEconomicaEmp_P_Educación',
           'SeccionEconomicaEmp_Q_Actv de atención de la Salud',
           'SeccionEconomicaEmp_R_Actv artisticas y Recreación',
           'SeccionEconomicaEmp_S_Otras Actividades de serv',
           'SeccionEconomicaEmp_T_Actv Hogares en calidad empl',
           'SeccionEconomicaEmp_U_Organiz Extraterritoriales',
           'TamanoEmp_Gran empresa', 'TamanoEmp_Mediana empresa',
           'TamanoEmp_Microempresa', 'TamanoEmp_No indica',
           'TamanoEmp_Pequeña empresa']

    # Reindex `data_num` to match `columns_final` and fill missing with 0
    expected_cols = list(scaler.feature_names_in_)  # columnas reales del scaler entrenado

    data_final_for_prediction = data_num.reindex(columns=expected_cols, fill_value=0)
    scaled_data = scaler.transform(data_final_for_prediction)

    # Prediction
    Y_pred = random_forest_model.predict(scaled_data)
    Y_pred_decoded = label_encoder.inverse_transform(Y_pred)

    st.success(f"¿Existe riesgo de que el afiliado y su grupo familiar dejen de usar los servicios ofrecidos por la caja? **{Y_pred_decoded}**")