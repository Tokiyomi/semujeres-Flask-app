#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Librerias utiles
import pandas as pd 
import numpy as np 

#import seaborn as sns 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#from mapsmx import MapsMX
#import geopandas
#import pyproj

import dash
from dash import dcc, callback
from dash import html
from dash.dependencies import Input, Output

import dash_bootstrap_components as dbc

from datetime import datetime,timedelta,date

#sns.set_theme()

#print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))


# In[ ]:

#external_stylesheets=[dbc.themes.LUMEN]
app = dash.Dash(__name__,suppress_callback_exceptions=True,)#external_stylesheets=external_stylesheets
server = app.server


# In[ ]:


"""centros = pd.read_excel('dbs/centros.xlsx')
centros=centros.loc[centros['FECHA DE BAJA'].isna()]
centros = centros[['CUENTA','NOMBRE','MUNICIPIO']]
centros = centros[centros.MUNICIPIO != 'TRANSVERSALIZACIÓN EN LA AP']
centros_no_duplicados = centros.drop_duplicates(subset=['CUENTA'],keep='last').reset_index(drop=True)
centros_no_duplicados = centros_no_duplicados.rename(columns={'MUNICIPIO':'MUNICIPIO CENTRO'})"""

import unicodedata

def strip_accents(text):
    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3 
        pass

    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")

    return str(text)

centros = pd.read_csv('dbs/BASE_CUENTAS_2022 (3).csv')
centros = centros.dropna(axis = 0, how = 'all')
centros['CRV ACTUAL']=centros['CRV ACTUAL'].fillna('DESCONOCIDO')
#centros['CRV ACTUAL'] = centros['CRV ACTUAL'].apply(lambda x: x.replace('2','x'))
#centros=centros.loc[centros['FECHA DE BAJA'].isna()]
centros = centros[['CUENTA','NOMBRE','CRV ACTUAL','FECHA DE ALTA','FECHA DE BAJA']]
centros['NOMBRE'] = centros.NOMBRE.apply(lambda x: x.strip())
centros['CRV ACTUAL'] = centros['CRV ACTUAL'].apply(lambda x: strip_accents(x))
#centros = centros[~(centros.MUNICIPIO.isin(['TRANSVERSALIZACIÓN EN LA AP']))]
centros_no_duplicados = centros.drop_duplicates(subset=['CUENTA'],keep='last').reset_index(drop=True)
centros_no_duplicados = centros_no_duplicados.rename(columns={'CRV ACTUAL':'MUNICIPIO CENTRO'})


# ## Load dfs and clean

# In[ ]:


"""feminicidios = pd.read_csv('dbs/feminicidios-corregida-03-marzo-2022.csv', low_memory=False)
feminicidios.fecha_recepcion = pd.to_datetime(feminicidios.fecha_recepcion, format='%d/%m/%Y', errors='ignore')
feminicidios['año_recepcion'] = feminicidios['fecha_recepcion'].dt.year
nas = feminicidios[(feminicidios.TipoRelacion.isna()==True)|(feminicidios.TipoRelacion=='Seleccione')].index
feminicidios.loc[nas,'TipoRelacion'] = 'Desconocido'
com = feminicidios[(feminicidios.TipoRelacion=='En la comunidad')].index
feminicidios.loc[com,'TipoRelacion'] = 'Comunidad'"""


# In[ ]:


servicios = pd.read_csv('dbs/servicios_29_AGOSTO_2022.csv', low_memory=False)
servicios.fecha_captura = pd.to_datetime(servicios.fecha_captura, format='%d/%m/%Y', errors='ignore')


# In[ ]:


status = pd.read_csv('dbs/status_29_AGOSTO_2022.csv', low_memory=False)
status['fecha_recepcion'] = pd.to_datetime(status.fecha_recepcion, format='%d/%m/%Y', errors='ignore')
status=status.loc[status.dependeciaUsuario.str.contains('MUJER')==True]
status=status.merge(centros_no_duplicados, left_on='fk_usuario', right_on='CUENTA', how='left')
status['año_recepcion'] = status.fecha_recepcion.dt.year
status.loc[status['MUNICIPIO CENTRO'].isna(),'MUNICIPIO CENTRO'] = 'DESCONOCIDO'
status=pd.DataFrame(status.groupby(by=['MUNICIPIO CENTRO','año_recepcion']).fk_euv.count()).rename(columns={'fk_euv':'Expedientes no enviados'})
status_tabla=status.reset_index()


# In[ ]:


victimas = pd.read_csv('reportes/reporte_semujeres_29_AGOSTO_2022.csv',low_memory=False, dtype={'pk_perfil_agresor': 'object','num_hijos':'int32'}, parse_dates=['fecha_recepcion','fecha_hechos'])


# In[ ]:


victimas.dropna(subset=['Dependencia de recepcion'],inplace=True)


# In[ ]:


victimas = pd.concat([victimas, victimas.tipos.str.get_dummies(sep=',')], axis=1)
victimas['Feminicida']=0
fems = victimas[victimas['descripcion_otro_tipos']=='FEMINICIDA'].index
victimas.loc[fems,'Feminicida'] = 1
#victimas.loc[fems,'Otro'] = 0
victimas.drop(columns='Otro', inplace=True)


# In[ ]:


victimas = pd.concat([victimas, victimas.ACTIVIDAD.str.get_dummies(sep=',')], axis=1)


# In[ ]:


validez = victimas[victimas.escolaridad=='Estudios que no requieren validéz oficial'].index
tecnica = victimas[victimas.escolaridad=='Carrera técnica comercial'].index
victimas.loc[validez,'escolaridad']='Sin validez'
victimas.loc[tecnica,'escolaridad']='Carrera técnica'


# In[ ]:


victimas['Dependencia de recepcion'] = victimas['Dependencia de recepcion'].str.slice(start=6)
juzgados = victimas[victimas['Dependencia de recepcion'].str.contains('TSJ-JUZGADO')==True].index
victimas.loc[juzgados,'Dependencia de recepcion']='PODER JUDICIAL'


# In[ ]:


"""no = victimas[victimas['Victima de Trata']=='Se desconoce'].index
victimas.loc[no, 'Victima de Trata']= 'No'


# In[ ]:


feminicidios = feminicidios.merge(victimas[['fk_euv','Edad Agresor']], left_on='fk_euv', right_on='fk_euv', how='left')
feminicidios = feminicidios.drop_duplicates(keep='last')
feminicidios = feminicidios.reset_index(drop=True)
fem_agrav = feminicidios[feminicidios.Sentencia.isin(['FEMINICIDIO AGRAVADO','FEMINICIDIO'])].index
feminicidios.loc[fem_agrav,'Sentencia']='FEMINICIDIO'
fem_agrav = feminicidios[feminicidios.Sentencia.isin(['FEMINICIDIO AGRAVADO EN GRADO DE TENTATIVA','FEMINICIDIO EN GRADO DE TENTATIVA'])].index
feminicidios.loc[fem_agrav,'Sentencia']='TENTATIVA DE FEMINICIDIO'


# In[ ]:


subtipo = pd.read_csv('dbs/subtipo_29_AGOSTO_2022.csv', low_memory=False)
subtipo.fecha_recepcion = pd.to_datetime(subtipo.fecha_recepcion, format='%d/%m/%Y', errors='ignore')
subtipo.fecha_hechos = pd.to_datetime(subtipo.fecha_hechos, format='%d/%m/%Y', errors='ignore')
subtipo = subtipo.dropna()
selecciones = subtipo[subtipo.SubtipoOrd=='Seleccione'].index
subtipo.loc[selecciones, 'SubtipoOrd'] = 'No especificado'"""


# In[ ]:


"""state = MapsMX().get_geo('state')
muns = MapsMX().get_geo('municipality')
yuc = muns[muns['cve_ent']=='31']


# In[ ]:


# censo poblacion y vivienda 2020 inegi
inegi_2020 = pd.read_csv('censo_yuc.csv')
pob_fem = inegi_2020[inegi_2020.NOM_LOC=='Total del Municipio'][['MUN','NOM_MUN','POBFEM' ]].reset_index(drop=True)


# In[ ]:


discapacidad = pd.read_csv('dbs/discapacidad.csv',low_memory=False)
discapacidad.fecha_recepcion = pd.to_datetime(discapacidad.fecha_recepcion, format='%d/%m/%Y', errors='ignore')
discapacidad = discapacidad.iloc[:, [0,1,3,22,60,61,62,63,64,65,66,67]]
discapacidad['Dependencia de recepcion'] = discapacidad['Dependencia de recepcion'].str.slice(start=6)
discapacidad = discapacidad.rename(columns={'\ncaminar, subir o bajar sus pies?': 'Motriz (piernas)', 'ver (aunque use lentes)?': 'Visual','mover o usar brazos o manos?*':'Motriz (brazos)',
'aprender, recordar o concentrarse?':'Intelectual','escuchar (aunque use aparato auditivo)?':'Auditiva','bañarse, vestirse o comer?':'Psicosocial (independencia)',
'hablar o comunicarse (por ejemplo, entender, o ser entendido por otros)?':'Psicosocial (comunicación)',
'or problemas emocionales o mentales, ¿Cuánta dificultad tiene la víctima para realizar sus actividades diarias (con autonomía e independencia)? Problemas como: autismo, depresión, bipolaridad, esquizofrenia,':'Psicosocial (mentales)'})
discapacidad.drop_duplicates(['fk_euv'],keep='last', inplace=True)
discapacidad.reset_index(drop=True, inplace=True)
discapacidad.replace(to_replace='No indicado ', value='No tiene dificultad', inplace=True)


# In[ ]:


discapacidad['Dependencia de recepcion']=discapacidad['Dependencia de recepcion'].fillna('DESCONOCIDA')
discapacidad.loc[discapacidad['Dependencia de recepcion'].str.contains('MUJER')==True,'Dependencia de recepcion']='SECRETARIA DE LAS MUJERES'


# In[ ]:


nan = victimas[victimas['Habla Indigena'].isna()==True].index
victimas.loc[nan, 'Habla Indigena'] = 'No'
des = victimas[victimas['Habla Indigena']=='Desconocido'].index
victimas.loc[des, 'Habla Indigena'] = 'No'


# In[ ]:


fiscalia = discapacidad[discapacidad['Dependencia de recepcion'].str.contains("SCALÍA")==True].index
mujer = discapacidad[discapacidad['Dependencia de recepcion'].str.contains("MUJER")==True].index
judicial = discapacidad[discapacidad['Dependencia de recepcion'].str.contains("JUDICIAL")==True].index
salud = discapacidad[discapacidad['Dependencia de recepcion'].str.contains("SALUD")==True].index
victimass = discapacidad[discapacidad['Dependencia de recepcion'].str.contains("VÍCTIMA")==True].index
seguridad = discapacidad[discapacidad['Dependencia de recepcion'].str.contains("SEGURIDAD")==True].index
juzgados = discapacidad[discapacidad['Dependencia de recepcion'].str.contains('TSJ-JUZGADO')==True].index
discapacidad.loc[fiscalia, 'Dependencia de recepcion'] = 'FISCALÍA GENERAL DEL ESTADO'
discapacidad.loc[mujer, 'Dependencia de recepcion'] = 'SECRETARÍA DE LAS MUJERES'
discapacidad.loc[judicial, 'Dependencia de recepcion'] = 'PODER JUDICIAL'
discapacidad.loc[salud, 'Dependencia de recepcion'] = 'SECRETARÍA DE SALUD'
discapacidad.loc[victimass, 'Dependencia de recepcion'] = ' COMISIÓN EJECUTIVA ESTATAL DE ATENCIÓN A VÍCTIMAS'
discapacidad.loc[seguridad, 'Dependencia de recepcion'] = 'SECRETARÍA DE SEGURIDAD PÚBLICA'
discapacidad.loc[juzgados,'Dependencia de recepcion']='PODER JUDICIAL'"""


# In[ ]:


servicios_semujeres = servicios[servicios['dependenciaquebrindoservicio']=='SECRETARIA DE LAS MUJERES']
servicios_semujeres = servicios_semujeres[['fk_euv','USUSERVICIO','fk_caso','fecha_captura','serviciodetalle','tiposervicio','estatus','dependenciaquebrindoservicio','numeroservicios','observaciones']]
servicios_semujeres.fk_caso = servicios_semujeres.fk_caso.astype(int)
servicios_semujeres.numeroservicios = servicios_semujeres.numeroservicios.astype(int)


# In[ ]:


servicios_semujeres['año_captura'] = servicios_semujeres.fecha_captura.dt.year
servicios_semujeres['mes_captura'] = servicios_semujeres.fecha_captura.dt.month


# In[ ]:


servicios_semujeres = servicios_semujeres[servicios_semujeres.año_captura>=2021]


# In[ ]:


"""canalizaciones = servicios_semujeres[(servicios_semujeres.observaciones.str.contains('CANALI'))&(~servicios_semujeres.observaciones.str.contains('JURIDICO'))].index
servicios_semujeres.loc[canalizaciones,'serviciodetalle'] = 'Canalización'"""

def normalize(s):
    replacements = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
    )
    for a, b in replacements:
        s = s.replace(a, b).replace(a.upper(), b.upper())
    return s

servicios_semujeres.observaciones = servicios_semujeres.observaciones.apply(lambda x: normalize(x))

canalizaciones = servicios_semujeres[(servicios_semujeres.observaciones.str.contains('CANALIZACION E|CANALIZACION PR|CANALIZACION.PR|CANALIZACION. PR'))&(servicios_semujeres.serviciodetalle!='Canalización')].index

extras = servicios_semujeres.loc[canalizaciones]

servicios_semujeres.loc[canalizaciones,'serviciodetalle'] = 'Canalización'

servicios_semujeres=servicios_semujeres.append(extras)

servicios_semujeres = servicios_semujeres.reset_index(drop=True)


# In[ ]:


#servicios_semujeres['semana']=servicios_semujeres.fecha_captura.dt.isocalendar().week


# In[ ]:


servicios_semujeres['semana']=servicios_semujeres.fecha_captura.dt.strftime('%W').astype(int)


# In[ ]:


servicios_semujeres = servicios_semujeres.merge(centros_no_duplicados, left_on='USUSERVICIO', right_on='CUENTA', how='left').drop(['CUENTA'],1)
servicios_semujeres['MUNICIPIO CENTRO'] = servicios_semujeres['MUNICIPIO CENTRO'].fillna('DESCONOCIDO')


# In[ ]:


idxs = servicios_semujeres[(servicios_semujeres.observaciones.str.contains('ENTREV'))&(servicios_semujeres.observaciones.str.contains('INIC'))&(servicios_semujeres.serviciodetalle=='Trabajo Social')&(~servicios_semujeres.observaciones.str.contains('SEGUI'))].index
servicios_semujeres.loc[idxs,'estatus']='Concluido'

idxs = servicios_semujeres[(servicios_semujeres.observaciones.str.contains('INTER'))&(servicios_semujeres.observaciones.str.contains('CRISIS'))].index
servicios_semujeres.loc[idxs,'estatus']='Concluido'

idxs = servicios_semujeres[(servicios_semujeres.observaciones.str.contains('CONTENC'))&(servicios_semujeres.observaciones.str.contains('EMOC'))].index
servicios_semujeres.loc[idxs,'estatus']='Concluido'

primera_vez_index = servicios_semujeres.sort_values(by=['fecha_captura','fk_euv','fk_caso']).drop_duplicates(subset=['fk_euv','año_captura'],keep='first').index
#primera_vez_index = servicios[servicios.numeroservicios==1].drop_duplicates(subset=['fk_euv','año_captura'],keep='first').index
servicios_semujeres['seguimiento']='Seguimiento'
servicios_semujeres.loc[primera_vez_index,'seguimiento']='Primera Vez'

# In[ ]:


# Esto tiene que ser exclusivamente por año
week_ranges = pd.DataFrame(servicios_semujeres.groupby([pd.Grouper(freq='W-SUN', key='fecha_captura')]).fk_euv.count()).reset_index().rename(columns={'fecha_captura':'fin_semana'})
week_ranges['inicio_semana'] = week_ranges['fin_semana'] - timedelta(days=6)
week_ranges['semana'] = week_ranges['fin_semana'].dt.strftime('%W').astype(int)
week_ranges['año'] = week_ranges['fin_semana'].dt.year
week_ranges=week_ranges.drop('fk_euv',axis=1)

servicios_semujeres=servicios_semujeres.merge(week_ranges, left_on=['año_captura','semana'], right_on=['año','semana'], how='left')

servicios_semujeres.loc[servicios_semujeres.fin_semana.isna(),['inicio_semana','fin_semana']] = [datetime(2021,12,27), datetime(2022,1,2)]


# In[ ]:


#servicios_semujeres.loc[(servicios_semujeres.semana==52)&(servicios_semujeres.año_captura==2021),'fin_semana'] = datetime(2021,12,31)
#servicios_semujeres.loc[(servicios_semujeres.semana==0)&(servicios_semujeres.año_captura==2021),'inicio_semana'] = datetime(2021,1,1)
#servicios_semujeres.loc[(servicios_semujeres.semana==0)&(servicios_semujeres.año_captura==2022),'inicio_semana'] = datetime(2022,1,1)


# In[ ]:


#semujeres_2021 = servicios_semujeres[servicios_semujeres['año_captura']==2021]


# In[ ]:


#barra = pd.DataFrame(semujeres_2021.groupby(by='mes_captura').count().numeroservicios)


# In[ ]:


#plt.bar(barra.index, barra.numeroservicios)


# In[ ]:


#barra = pd.DataFrame(semujeres_2021.drop_duplicates(subset=['mes_captura','fk_euv']).groupby(by='mes_captura').count().fk_euv)


# In[ ]:


#plt.bar(barra.index, barra.fk_euv)


# In[ ]:


#semujeres_2021.drop_duplicates(subset=['mes_captura','fk_euv']).groupby(by='mes_captura').count().fk_euv.mean()


# # Dashboard

# In[ ]:


# Menu
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

index_page = html.Div([
    dcc.Link('Go to Page 1', href='/page-1'),
    html.Br(),
    dcc.Link('Go to Page 2', href='/page-2'),
])

"""page_1_layout = html.Div(
    children=[
        html.H1(children="SEMUJERES TABLERO",),
        html.P(
            children="Texto de prueba",
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(children="Dependencia de recepcion", className="menu-title"),
                        dcc.Dropdown(
                            id="dependencia-filter",
                            options=[
                                {"label": dep, "value": dep}
                                for dep in (victimas['Dependencia de recepcion'].unique())
                            ],
                            value='SECRETARIA DE LAS MUJERES',
                            multi=False,
                            clearable=False,
                            #className="dropdown",
                        ),
                    ]
                ),
                #html.Div(
                    #children=[
                        #html.Div(children="Municipio", className="menu-title"),
                        #dcc.Dropdown(
                            #id="municipio-filter",
                            #options=[
                                #{"label": mun, "value": mun}
                                #for mun in victimas.municipiohechos.unique()
                            #],
                            #value=[mun for mun in victimas.municipiohechos.unique()],
                            #multi=True,
                            #clearable=True,
                            #searchable=True,
                            #className="dropdown",
                        #),
                    #],
                #),
                html.Div(
                    children=[
                        html.Div(
                            children="Date Range",
                            className="menu-title"
                            ),
                        dcc.DatePickerRange(
                            id="date-range",
                            min_date_allowed=victimas.fecha_recepcion.min(),
                            max_date_allowed=victimas.fecha_recepcion.max(),
                            start_date=victimas.fecha_recepcion.min(),
                            end_date=victimas.fecha_recepcion.max(),
                        ),
                    ]
                ),
            ],
            className="menu",
        ),
        dcc.Graph(
            id = 'map_1',  
        ),
        dcc.Graph(
            id = 'map_2',  
        ),
        dcc.Graph(
            id = 'fig_1',  
        ),
        dcc.Graph(
            id = 'fig_2',  
        ),
        dcc.Graph(
            id = 'fig_3',  
        ),
        dcc.Graph(
            id = 'fig_8',  
        ),
        dcc.Graph(
            id = 'fig_4',  
        ),
        dcc.Graph(
            id = 'fig_5',  
        ),
        dcc.Graph(
            id = 'fig_6',  
        ),
        dcc.Graph(
            id = 'fig_7',  
        ),
        dcc.Graph(
            id = 'fig_9',  
        ),
        dcc.Graph(
            id = 'fig_10',  
        ),
        dcc.Graph(
            id = 'fig_11',  
        ),
        dcc.Graph(
            id = 'fig_12',  
        ),
        dcc.Graph(
            id = 'fig_13',  
        ),
        dcc.Graph(
            id = 'fig_14',  
        ),
        dcc.Graph(
            id = 'fig_15',  
        ),
        html.Br(),
        dcc.Link('Go to Page 2', href='/page-2'),
        html.Br(),
        dcc.Link('Go back to home', href='/'),
    ]
)

@callback(
    [   Output('fig_1', "figure"),
        Output('map_1', "figure"),
        Output('fig_2', "figure"),
        Output('map_2', "figure"),
        Output('fig_3', "figure"),
        Output('fig_4', "figure"),
        Output('fig_5', "figure"),
        Output('fig_6', "figure"),
        Output('fig_7', "figure"),
        Output('fig_8', "figure"),
        Output('fig_9', "figure"),
        Output('fig_10', "figure"),
        Output('fig_11', "figure"),
        Output('fig_12', "figure"),
        Output('fig_13', "figure"),
        Output('fig_14', "figure"),
        Output('fig_15', "figure"),
    ],
    [
        Input("dependencia-filter", "value"),
        #Input("municipio-filter", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
    ]
)
def update_charts(dependencia, start_date, end_date):
    mask = (
        (victimas['Dependencia de recepcion'] == dependencia)
        #& (victimas['municipiohechos'].isin(municipio) ==True)
        & (victimas.fecha_recepcion >= start_date)
        & (victimas.fecha_recepcion <= end_date)
    )

    filtered_data = victimas.loc[mask, :]
    
    # Fig 1 - tipos de violencia
    tipos = filtered_data.iloc[:,[i for i in range(72,78)]]
    tipos = pd.DataFrame(tipos.sum())
    tipos = tipos.sort_values(0,ascending=False)
    y = tipos[0].values
    y_total = len(filtered_data)

    fig_1 = px.bar(x=tipos.index,
                y=tipos[0],
                text= np.round(y/y_total*100,2),
                labels = {'x': 'Tipo de violencia', "y":'Número de casos', 'text':'Porcentaje'},
                color=px.colors.qualitative.Prism[:6],
                #color=px.colors.sequential.matter[::-1][:6][::],
                color_discrete_map="identity",
                title='Gráfica 1. Casos de violencia hacia las mujeres por tipo de violencia'
                )
    fig_1.update_xaxes(type='category')
    fig_1.update_layout( xaxis_title=None)
    fig_1.update_traces(texttemplate='%{text} %')
    fig_1.update_traces(hovertemplate='Tipo de violencia=%{x}<br>Número de casos=%{y}<br>Porcentaje=%{text} %<extra></extra>')

    # Map 1 - casos x mun
    gdf = geopandas.GeoDataFrame(yuc, geometry='geometry_mun')
    
    mun_counts = pd.DataFrame(filtered_data.municipiohechos.value_counts())
    mun_counts.reset_index(inplace=True)
    
    gdf = pd.DataFrame(gdf.merge(mun_counts,left_on='nom_mun',right_on='index',how='left'))
    gdf = geopandas.GeoDataFrame(gdf, geometry='geometry_mun')
    gdf = gdf.set_index('nom_mun')
    gdf = gdf.rename(columns={'municipiohechos':'Casos registrados'})
    gdf['Casos registrados log'] = np.log10(gdf['Casos registrados'] )
    gdf_crs = gdf.to_crs(pyproj.CRS.from_epsg(4326))
    #gdf_crs = gdf_crs.fillna(0)

    map_1 = px.choropleth(gdf_crs, geojson=gdf_crs.geometry_mun, 
                    locations=gdf_crs.index, color="Casos registrados log",
                    height=500,
                   color_continuous_scale="Viridis",
                    labels = {'nom_mun':'Municipio', "Casos registrados log":'log10'},
                    hover_data={"Casos registrados":True, "Casos registrados log":False},
                    title='Mapa 1. Casos Registrados por Municipio',
                   )
    map_1.update_geos(fitbounds="locations", visible=False)

    map_1.update_layout(xaxis=dict(domain=[0, 0.5]), yaxis=dict(domain=[0.25, 0.75]))

    map_1.add_annotation(
        # The arrow head will be 25% along the x axis, starting from the left
        x=0,
        # The arrow head will be 40% along the y axis, starting from the bottom
        y=0.98,
        text="<b>Temporalidad:</b> {}-{}<br><b>Total de casos:</b> {}".format(min(filtered_data.año_recepcion),max(filtered_data.año_recepcion),len(filtered_data)),
        showarrow=False,
        bordercolor="black",
        bgcolor="white",
        borderwidth=1.5,
        opacity=0.8
    )

    map_1.update(layout = dict(title=dict(x=0.5)))
    map_1.update_layout(
        margin={"r":0,"t":30,"l":10,"b":10},
        coloraxis_colorbar={
            'title':'Escala'})
    map_1.update_layout(title_y=1, title_x=0)
    map_1.update_layout(coloraxis_showscale=True)

    # Map 2 - Tasa 1000
    dep = victimas[victimas['Dependencia de recepcion']==dependencia]

    mun_counts = pd.DataFrame(dep.municipiohechos.value_counts())
    mun_counts.reset_index(inplace=True)

    dep_2020 = victimas[victimas['Dependencia de recepcion']==dependencia]
    dep_2020 = dep_2020[dep_2020.año_recepcion==2020]

    mun_counts_2020 = pd.DataFrame(dep_2020.municipiohechos.value_counts())
    mun_counts_2020.reset_index(inplace=True)

    gdf = geopandas.GeoDataFrame(yuc, geometry='geometry_mun')
    gdf_2020 = pd.DataFrame(gdf.merge(mun_counts_2020,left_on='nom_mun',right_on='index',how='left'))
    gdf_2020 = pd.DataFrame(gdf_2020.merge(pob_fem,left_on='nom_mun',right_on='NOM_MUN',how='left'))
    gdf_2020 = geopandas.GeoDataFrame(gdf_2020, geometry='geometry_mun')
    gdf_2020 = gdf_2020.set_index('nom_mun')
    gdf_2020 = gdf_2020.rename(columns={'municipiohechos':'Casos registrados'})
    gdf_2020['Tasa 1000 casos'] = round(gdf_2020['Casos registrados']/gdf_2020['POBFEM'].astype(int)*1000,2)
    gdf_crs_2020 = gdf_2020.to_crs(pyproj.CRS.from_epsg(4326))
    gdf_crs_2020['Tasa 1000 casos'] = gdf_crs_2020['Tasa 1000 casos'].fillna(0)
    gdf_crs_2020['Casos registrados'] = gdf_crs_2020['Casos registrados'].fillna(0)

    map_2 = px.choropleth(gdf_crs_2020, geojson=gdf_crs_2020.geometry_mun, 
                    locations=gdf_crs_2020.index, color="Tasa 1000 casos",
                    #height=500,
                    #color_continuous_scale="Viridis",
                    #color_continuous_midpoint=np.average(gdf_crs_2020['Tasa 1000 casos'], weights=gdf_crs_2020['POBFEM'].astype(int)),
                    labels = {'nom_mun':'Municipio', 'POBFEM':'Población Femenina del Municipio', 'Tasa 1000 casos':'Tasa'},
                    hover_data={"Casos registrados":True,"POBFEM":True,},
                    title='Mapa 2. Tasa de casos registrados por cada 1000 mujeres por Municipio en 2020',
                    )
    map_2.update_geos(fitbounds="locations", visible=False)

    map_2.update_layout(xaxis=dict(domain=[0, 0.5]), yaxis=dict(domain=[0.25, 0.75]))

    map_2.add_annotation(
        # The arrow head will be 25% along the x axis, starting from the left
        x=0,
        # The arrow head will be 40% along the y axis, starting from the bottom
        y=0.98,
        text="<b>Temporalidad:</b> {}<br><b>Total de casos:</b> {}".format(2020,len(dep_2020)),
        showarrow=False,
        bordercolor="black",
        bgcolor="white",
        borderwidth=1.5,
        opacity=0.8
    )

    map_2.update(layout = dict(title=dict(x=0.5)))
    map_2.update_layout(
        margin={"r":0,"t":30,"l":10,"b":10},
        coloraxis_colorbar={
            'title':'Tasa'})
    map_2.update_layout(title_y=1, title_x=0)
    map_2.update_layout(coloraxis_showscale=True)

    # fig 2 - estudios
    y_2 = filtered_data.escolaridad.value_counts().values
    y_total_2 = sum(y_2)
    colors = len(y_2)
    fig_2 = px.bar(x=filtered_data.escolaridad.value_counts().index,
                    y=filtered_data.escolaridad.value_counts().values,
                    text= np.round(y_2/y_total_2*100,2),
                    labels = {'x': 'Escolaridad', "y":'Número de casos', 'text':'Porcentaje'},
                    color=px.colors.qualitative.Prism[:colors],
                    color_discrete_map="identity",
                    title='Gráfica 2. Estudios Concluidos'
                    )
    fig_2.update_xaxes(type='category')
    fig_2.update_layout(xaxis_title=None)
    fig_2.update_traces(texttemplate='%{text} %')

    # Fig 3 - Civil
    y = filtered_data.EstadoCivil.value_counts().values
    y_total = sum(y)
    colors = len(y)
    fig_3 = px.bar(x=filtered_data.EstadoCivil.value_counts().index,
                y=filtered_data.EstadoCivil.value_counts().values,
                text= np.round(y/y_total*100,2),
                labels = {'x': 'Estado Civil', "y":'Número de casos', 'text':'Porcentaje'},
                color=px.colors.qualitative.Prism[:colors],
                color_discrete_map="identity",
                title='Gráfica 3. Estado Civil'
                )
    fig_3.update_xaxes(type='category')
    fig_3.update_layout(xaxis_title=None)
    fig_3.update_traces(texttemplate='%{text} %')

    # Fig 4 - Discapacidad
    if dependencia in discapacidad['Dependencia de recepcion'].unique():
        discapacidades_tabla = pd.DataFrame()
        mask_dis = (
        (discapacidad['Dependencia de recepcion'] == dependencia)
        & (discapacidad.fecha_recepcion >= start_date)
        & (discapacidad.fecha_recepcion <= end_date)
        )
        discapacidad_dep = discapacidad.loc[mask_dis, :]
        discapacidad_tipos = discapacidad_dep.iloc[:, 4:]
        if len(discapacidad_tipos)!=0:
            for col in discapacidad_tipos.columns:
                new_df = pd.DataFrame(discapacidad_tipos[col].value_counts()).unstack().reset_index()
                discapacidades_tabla=discapacidades_tabla.append(new_df, ignore_index=True)
            discapacidades_tabla = discapacidades_tabla.rename(columns={'level_0':'Tipo de Discapacidad','level_1':'Nivel de dificultad',0:'Cuenta'})
            discapacidades_tabla.replace(to_replace='Motriz (brazos)', value='Motriz', inplace=True)
            discapacidades_tabla.replace(to_replace='Motriz (piernas)', value='Motriz', inplace=True)
            discapacidades_tabla.replace(to_replace='Psicosocial (comunicación)', value='Psicosocial', inplace=True)
            discapacidades_tabla.replace(to_replace='Psicosocial (independencia)', value='Psicosocial', inplace=True)
            discapacidades_tabla.replace(to_replace='Psicosocial (mentales)', value='Psicosocial', inplace=True)
            discapacidades_tabla['Porcentaje']=round(discapacidades_tabla.Cuenta/len(discapacidad_tipos)*100,2)
            discapacidades_tabla = discapacidades_tabla[discapacidades_tabla['Nivel de dificultad']!='No tiene dificultad']
            total_mujeres = pd.DataFrame(discapacidades_tabla.groupby('Tipo de Discapacidad').Cuenta.sum()).sort_values('Cuenta', ascending=False)
            discapacidades_tabla=discapacidades_tabla.groupby(['Tipo de Discapacidad','Nivel de dificultad']).sum().reset_index()
            discapacidades_tabla['Tipo de Discapacidad'] = pd.Categorical(discapacidades_tabla['Tipo de Discapacidad'], total_mujeres.index)
            discapacidades_tabla= discapacidades_tabla.sort_values('Tipo de Discapacidad').reset_index(drop=True)
            
            fig_4 = px.bar(discapacidades_tabla, x="Tipo de Discapacidad", y="Porcentaje", color="Nivel de dificultad", 
            title="Gráfica 5. Número de mujeres con discapacidad",
            hover_name='Tipo de Discapacidad',
            text= "Cuenta",
            hover_data={"Cuenta":True, 'Tipo de Discapacidad':False},
            labels = {'Cuenta': 'Número de mujeres'},
            color_discrete_sequence=px.colors.qualitative.Prism[:5])
            #fig_4.update_layout(height=500)
            fig_4.update_traces(textposition='outside', cliponaxis=False)
            fig_4.update_layout(yaxis_ticksuffix = '%')
        else:
            fig_4=px.bar(title="Gráfica 5. Número de mujeres con discapacidad: SIN INFORMACION EN ESTAS FECHAS") 
    else:
        fig_4=px.bar(title="Gráfica 5. Número de mujeres con discapacidad: SIN INFORMACION PARA ESTA DEPENDENCIA")
        
    # Fig 5 - Indigena
    y = filtered_data['Habla Indigena'].value_counts().values
    y_total = sum(y)
    prismas = px.colors.qualitative.Prism 
    prismas += px.colors.qualitative.Prism
    colors=len(y)
    fig_5 = px.bar(x=filtered_data['Habla Indigena'].value_counts().index,
                y=filtered_data['Habla Indigena'].value_counts().values,
                text= np.round(y/y_total*100,2),
                labels = {'x': 'Habla lengua indígena', "y":'Número de casos', 'text':'Porcentaje'},
                color=prismas[:colors],
                color_discrete_map="identity",
                title='Gráfica 6. Mujeres que hablan Lengua Indígena'
                )
    fig_5.update_xaxes(type='category')
    fig_5.update_layout(xaxis_title=None, height=500)
    fig_5.update_traces(texttemplate='%{text} %')

    # Fig 6 - Conocimiento autoridad
    y = filtered_data['Conocimiento de autoridad'].value_counts().values
    y_total = sum(y)
    colors = len(y)
    fig_6 = px.bar(x=filtered_data['Conocimiento de autoridad'].value_counts().index,
                y=filtered_data['Conocimiento de autoridad'].value_counts().values,
                text= np.round(y/y_total*100,2),
                labels = {'x': 'Conocimiento de alguna autoridad', "y":'Número de casos', 'text':'Porcentaje'},
                color=px.colors.qualitative.Prism[:colors],
                color_discrete_map="identity",
                title='Gráfica 7. Casos que son conocidos por alguna autoridad'
                )
    fig_6.update_xaxes(type='category')
    fig_6.update_layout( xaxis_title=None)
    fig_6.update_traces(texttemplate='%{text} %')

    # Fig 7 - Autoridad conoce caso
    si_conocimiento = filtered_data[filtered_data['Conocimiento de autoridad']=='Si']
    mp = si_conocimiento['Descripcion de autoridad'].str.contains("MINISTERIO").sum() + si_conocimiento['Descripcion de autoridad'].str.contains("MP").sum() + si_conocimiento['Descripcion de autoridad'].str.contains("M.P").sum() 
    fg = si_conocimiento['Descripcion de autoridad'].str.contains("FIS").sum() + si_conocimiento['Descripcion de autoridad'].str.contains("FG").sum() + si_conocimiento['Descripcion de autoridad'].str.contains("F.G").sum()
    juz = si_conocimiento['Descripcion de autoridad'].str.contains("JUZ").sum() + si_conocimiento['Descripcion de autoridad'].str.contains("JUEZ").sum() + si_conocimiento['Descripcion de autoridad'].str.contains("JZ").sum()
    agencia = si_conocimiento['Descripcion de autoridad'].str.contains("AGE").sum() + si_conocimiento['Descripcion de autoridad'].str.contains("AG").sum()
    cjm = si_conocimiento['Descripcion de autoridad'].str.contains("JUS").sum() + si_conocimiento['Descripcion de autoridad'].str.contains("CJ").sum() + si_conocimiento['Descripcion de autoridad'].str.contains("JM").sum() + si_conocimiento['Descripcion de autoridad'].str.contains("B1").sum()
    prod = si_conocimiento['Descripcion de autoridad'].str.contains("PRODE").sum()
    salud = si_conocimiento['Descripcion de autoridad'].str.contains("SALUD").sum() + si_conocimiento['Descripcion de autoridad'].str.contains("HOSP").sum()
    judicial = si_conocimiento['Descripcion de autoridad'].str.contains("MUN").sum() + si_conocimiento['Descripcion de autoridad'].str.contains("SSP").sum()
    judicial_juzgado = judicial+juz
    fis_agencia = fg + agencia
    dic_autoridades = { 'Autoridad':['Ministerio Público','Fiscalía','PRODENNAY', 'CJM','Centro de Salud', 'Poder Judicial'],
                    #'Porcentaje':100*np.array([mp, fis_agencia,  prod,  cjm, salud,judicial_juzgado])/sum([mp, fis_agencia,  prod,  cjm, salud, judicial_juzgado]),
                    'Porcentaje':100*np.array([mp, fis_agencia,  prod,  cjm, salud,judicial_juzgado])/len(si_conocimiento),
                    'Count': np.array([mp, fis_agencia, prod,  cjm, salud, judicial_juzgado])}
    autoridades = pd.DataFrame(dic_autoridades)
    autoridades = autoridades.sort_values('Count', ascending=False)
    y = autoridades['Count']
    y_total = len(si_conocimiento)
    fig_7 = px.bar(x=autoridades['Autoridad'],
                y=autoridades['Count'],
                text= np.round(y/y_total*100,2),
                labels = {'x': 'Autoridad', "y":'Número de casos', 'text':'Porcentaje'},
                color=px.colors.qualitative.Prism[:6],
                color_discrete_map="identity",
                title='Gráfica 8. Autoridad que conoce el caso'
                )
    fig_7.update_xaxes(type='category')
    fig_7.update_layout( xaxis_title=None,)
    fig_7.update_traces(texttemplate='%{text} %')

    # Fig 8 - Actividad que realizan las mujeres
    y = [filtered_data['Trabaja en el hogar'].sum(), filtered_data['Trabaja fuera del hogar'].sum(), filtered_data['Estudia'].sum(), filtered_data['Jubilada/Pensionada'].sum()+filtered_data['Pensionada'].sum(),filtered_data['Otro'].sum(),filtered_data['Se desconoce'].sum()]
    #y_total = sum(y)
    y_total = np.array(len(filtered_data), dtype=int)
    fig_8 = px.bar(x=['Trabaja en el hogar', 'Trabaja fuera del hogar','Estudia','Jubilada/Pensionada','Otro','Se desconoce'], 
                y = y,
                text= np.round(y/y_total*100,2),
                labels = {'x': 'Actividad', "y":'Número de casos', 'text':'Porcentaje'},
                color=px.colors.qualitative.Prism[:6],
                color_discrete_map="identity",
                title='Gráfica 4. Actividad que realizan las mujeres'
                )
    fig_8.update_xaxes(type='category')
    fig_8.update_layout( xaxis={'categoryorder':'total descending'}, xaxis_title=None,)
    #fig.update_traces(texttemplate='%{text} %', textposition='outside')
    fig_8.update_traces(texttemplate='%{text} %')

    # Fig 9 - Modalidad
    y = filtered_data.modalidad.value_counts().values
    y_total = sum(y)
    colors = len(y)
    fig_9 = px.bar(x=filtered_data.modalidad.value_counts().index,
                y=filtered_data.modalidad.value_counts().values,
                text= np.round(y/y_total*100,2),
                labels = {'x': 'Modalidad de la violencia', "y":'Número de casos', 'text':'Porcentaje'},
                color=px.colors.qualitative.Prism[:colors],
                color_discrete_map="identity",
                title='Gráfica 9. Modalidad de la violencia'
                )
    fig_9.update_xaxes(type='category')
    fig_9.update_layout( xaxis_title=None,)
    fig_9.update_traces(texttemplate='%{text} %')

    # Fig 10 - vinculo victima
    y = filtered_data['Tipo de vínculo con victima'].value_counts().values
    y_total = sum(y)
    colors = len(y)
    fig_10 = px.bar(x=filtered_data['Tipo de vínculo con victima'].value_counts().index,
                y=filtered_data['Tipo de vínculo con victima'].value_counts().values,
                text= np.round(y/y_total*100,2),
                labels = {'x': 'Tipo de vínculo con la víctima', "y":'Número de casos', 'text':'Porcentaje'},
                color=px.colors.qualitative.Prism[:colors],
                color_discrete_map="identity",
                title='Gráfica 10. Tipo de vínculo con la víctima'
                )
    fig_10.update_xaxes(type='category')
    fig_10.update_layout( xaxis_title=None)
    fig_10.update_traces(texttemplate='%{text} %')

    # Fig 11 - detalle vinculo victima
    y = filtered_data['Detalle del Tipo Vínculo con victima'].value_counts().values
    y_total = sum(y) 
    colors = len(y)
    fig_11 = px.bar(x=filtered_data['Detalle del Tipo Vínculo con victima'].value_counts().index,
                y=filtered_data['Detalle del Tipo Vínculo con victima'].value_counts().values,
                text= np.round(y/y_total*100,2),
                labels = {'x': 'Detalle del vínculo con la víctima', "y":'Número de casos', 'text':'Porcentaje'},
                color=prismas[:colors],
                color_discrete_map="identity",
                title='Gráfica 11. Detalle del vínculo con la víctima'
                )
    fig_11.update_xaxes(type='category')
    fig_11.update_layout( xaxis={'categoryorder':'total descending'})
    fig_11.update_layout( xaxis_title=None)
    fig_11.update_traces(texttemplate='%{text} %')

    # Fig 12 - Violencia ultimo año
    tuvo_violencia = filtered_data[(filtered_data['Detalle del Tipo Vínculo con victima'].isin(['Cónyuge o pareja ', 'Ex pareja']))&(filtered_data['fecha_recepcion']>=(filtered_data.fecha_recepcion.max()-pd.offsets.DateOffset(years=1)))]
    ultimo_año = filtered_data[filtered_data['fecha_recepcion']>=(filtered_data.fecha_recepcion.max()-pd.offsets.DateOffset(years=1))]
    ultimo_año['violencia_ex_pareja'] = 'No'
    ultimo_año.loc[tuvo_violencia.index, 'violencia_ex_pareja'] = 'Si'
    y = ultimo_año['violencia_ex_pareja'].value_counts().values
    y_total = len(ultimo_año)
    colors = len(y)
    fig_12 = px.bar(x=ultimo_año['violencia_ex_pareja'].value_counts().index,
                y=ultimo_año['violencia_ex_pareja'].value_counts().values,
                text= np.round(y/y_total*100,2),
                labels = {'x': 'Victima de violencia', "y":'Número de casos', 'text':'Porcentaje'},
                color=px.colors.qualitative.Prism[:colors],
                color_discrete_map="identity",
                title=""Gráfica 12. Mujeres que han experimentado violencia por parte de su última pareja en los últimos 12 meses<br><b>Total de casos en los últimos 12 meses:</b> {}"".format(len(ultimo_año))
                )
    fig_12.update_xaxes(type='category')
    fig_12.update_layout( xaxis_title=None,)
    fig_12.update_traces(texttemplate='%{text} %')

    # Fig 13 - Victima de trata
    y = filtered_data['Victima de Trata'].value_counts().values
    y_total = sum(y)
    colors = len(y)
    fig_13 = px.bar(x= filtered_data['Victima de Trata'].value_counts().index,
                y= filtered_data['Victima de Trata'].value_counts().values,
                text= np.round(y/y_total*100,2),
                labels = {'x': 'Victima de Trata', "y":'Número de casos', 'text':'Porcentaje'},
                color=px.colors.qualitative.Prism[:colors],
                color_discrete_map="identity",
                title='Gráfica 13. Casos relacionados con el delito de Trata'
                )
    fig_13.update_xaxes(type='category')
    fig_13.update_layout( xaxis_title=None)
    fig_13.update_traces(texttemplate='%{text} %')

    # Fig 14 - Delincuencia\
    y = filtered_data['Victima de delincuencia'].value_counts().values
    y_total = sum(y)
    colors=len(y)
    fig_14 = px.bar(x=filtered_data['Victima de delincuencia'].value_counts().index,
                y=filtered_data['Victima de delincuencia'].value_counts().values,
                text= np.round(y/y_total*100,2),
                labels = {'x': 'Victima de delincuencia', "y":'Número de casos', 'text':'Porcentaje'},
                color=px.colors.qualitative.Prism[:colors],
                color_discrete_map="identity",
                title='Gráfica 14. Casos relacionados con Delincuencia Organizada'
                )
    fig_14.update_xaxes(type='category')
    fig_14.update_layout( xaxis_title=None)
    fig_14.update_traces(texttemplate='%{text} %')
    #fig.update_xaxes(visible=False, showticklabels=False)
    fig_14.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = [0, 1],
            ticktext = ['No','Si']
        )
    )

    # Fig 15 - Servicios brindados
    if dependencia in servicios['dependenciaquebrindoservicio'].unique():
        mask_ser = (
        (servicios['dependenciaquebrindoservicio'] == dependencia)
        & (servicios.fecha_captura >= start_date)
        & (servicios.fecha_captura <= end_date)
        )
        servicios_data = servicios.loc[mask_ser, :]
        if len(servicios_data)!=0:
            y = servicios_data.serviciodetalle.value_counts().values
            y_total = sum(y)
            colors = len(y)
            fig_15 = px.bar(x=servicios_data.serviciodetalle.value_counts().index,
                        y=servicios_data.serviciodetalle.value_counts().values,
                        text= np.round(y/y_total*100,2),
                        labels = {'x': 'Servicio proporcionado', "y":'Número de casos', 'text':'Porcentaje'},
                        color=prismas[:colors],
                        color_discrete_map="identity",
                        title='Gráfica 15. Servicios proporcionados por la dependencia'
                        )
            fig_15.update_xaxes(type='category')
            fig_15.update_layout( xaxis={'categoryorder':'total descending'},)
            fig_15.update_layout( xaxis_title=None)
            fig_15.update_traces(texttemplate='%{text} %')
        else:
            fig_15=px.bar(title='Gráfica 15. Servicios proporcionados por la dependencia: SIN INFORMACION PARA ESTAS FECHAS')
    else:
        fig_15=px.bar(title='Gráfica 15. Servicios proporcionados por la dependencia: SIN INFORMACION PARA ESTA DEPENDENCIA')
    
    return fig_1,map_1,fig_2,map_2,fig_3,fig_4,fig_5, fig_6, fig_7, fig_8,fig_9,fig_10,fig_11,fig_12, fig_13, fig_14, fig_15"""

"""page_2_layout = html.Div([
    html.H1('Page 2'),
    dcc.RadioItems(options=['Orange', 'Blue', 'Red'], value='Orange', id='page-2-radios'),
    html.Div(id='page-2-content'),
    html.Br(),
    dcc.Link('Go to Page 1', href='/page-1'),
    html.Br(),
    dcc.Link('Go back to home', href='/')
])"""

page_2_layout = html.Div(
    children=[
        
                #html.H1(children="Tablero de Control de los Servicios de Atención a la Violencia",className="header",),
                #html.P(
                    #children="Texto de prueba",
                #),
                html.Div(
                children=[
                    html.P(children="🚨", className="header-emoji"),
                    #html.Img(src='/assets/yuc_logo.png', className="header-img"),
                    html.H1(
                        children="Tablero de Control de los Servicios de Atención a la Violencia", className="header-title"
                    ),
                    html.P( 
                        children="Última Actualización: {}".format(pd.to_datetime(str(servicios_semujeres.fecha_captura.max())).strftime('%d-%m-%Y')),
                        className="header-description",
                    ),
                ],
                className="header",
                ),
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        html.Div(
                                            children="Seleccione el rango de fechas",
                                            className="menu-title"
                                            ),
                                        dcc.DatePickerRange(
                                            id="date-range-2",
                                            min_date_allowed=date(2021,1,1),
                                            max_date_allowed=date(servicios_semujeres.fecha_captura.max().year,servicios_semujeres.fecha_captura.max().month,servicios_semujeres.fecha_captura.max().day),
                                            start_date=date(2022,1,1),
                                            end_date=date(servicios_semujeres.fecha_captura.max().year,servicios_semujeres.fecha_captura.max().month,servicios_semujeres.fecha_captura.max().day),
                                            display_format='DD/MM/Y',
                                        ),
                                    ]
                                ),
                            ],
                            className="menu",
                            ),
                        ],),
                        #html.Div(
                            #children="Inserte Año",
                            #className="menu-title"
                            #),
                        #dcc.DatePickerRange(
                            #id="date-range-2",
                            #min_date_allowed=servicios_semujeres.fecha_captura.min(),
                            #max_date_allowed=servicios_semujeres.fecha_captura.max(),
                            #start_date=servicios_semujeres.fecha_captura.min(),
                            #end_date=servicios_semujeres.fecha_captura.max(),
                        #),
                        #dcc.Dropdown(
                            #id="year-2",
                            #options=[2021,2022],
                            #value=2022,
                            #clearable=False
                        #),
                        #html.Br(),
                        #html.Div(
                            #children="Seleccione las semanas (cada semana se contabiliza de lunes a domingo, la semana 1 empieza con el primer lunes del año)",
                            #className="menu-title"
                            #),
                        #dcc.Input(id='semana', type='number', min=0, max=53,value=None, debounce=False),
                        #dcc.RangeSlider(0, 52, 1, value=[0, servicios_semujeres[servicios_semujeres.año_captura==2022].semana.max()], id='my-range-slider'),
                        #html.Div(id='rango_semanas')
                        #html.Div(
                            #children="Seleccione un rango de fechas",
                            #),
                        #dcc.DatePickerRange(
                            #id="date-range-2",
                            #min_date_allowed=date(2021,1,1),
                            #max_date_allowed=date(servicios_semujeres.fecha_captura.max().year,servicios_semujeres.fecha_captura.max().month,servicios_semujeres.fecha_captura.max().day),
                            #start_date=date(2022,1,1),
                            #end_date=date(servicios_semujeres.fecha_captura.max().year,servicios_semujeres.fecha_captura.max().month,servicios_semujeres.fecha_captura.max().day),
                            #display_format='DD/MM/Y',
                        #),
                    #],
                    #className="menu",
                #),
            
        #dcc.Graph(
            #id = 'fig_11_2',  
        #),
        # wrapper
        html.Div(children=[
        # Agregado pa prueba, sacar grapsh 1 y 2 despues
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(html.Div(children=[dcc.Graph(id = 'fig_11_2',)],className="card"),width=12),
                    ],
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Div(children=[dcc.Graph(id = 'fig_14_2',),],className="card",),width=12),
                    ], 
                ),
                dbc.Row(
            [
                dbc.Col(html.Div(children=[dcc.Graph(id = 'fig_1_2',)],className="card",),width=6),
                dbc.Col(html.Div(children=[dcc.Graph(id = 'fig_2_2',)],className="card",),width=6),
                
            ], 
        ),
                dbc.Row(
                    [
                        dbc.Col(html.Div(children=[dcc.Graph(id = 'fig_3_2',)],className="card",),width=6),
                        dbc.Col(html.Div(children=[dcc.Graph(id = 'fig_4_2',)],className="card",),width=6),
                    ], 
                ),
            ], fluid=True
        ),
        #dcc.Graph(
            #id = 'fig_1_2',  
        #),
        #dcc.Graph(
            #id = 'fig_2_2',  
        #),
        #dcc.Graph(
            #id = 'fig_3_2',  
        #),
        #dcc.Graph(
            #id = 'fig_4_2',  
        #),
        #dcc.Graph(
            #id = 'fig_5_2',  
        #),
        dbc.Container(
            [
                dbc.Row(
            [
                dbc.Col(html.Div(children=[dcc.Graph(id = 'fig_6_2',),],className="card",),width=6),
                dbc.Col(html.Div(children=[dcc.Graph(id = 'fig_7_2',),],className="card",),width=6),
                
            ], 
        ),
                dbc.Row(
                    [
                        dbc.Col(html.Div(children=[dcc.Graph(id = 'fig_8_2',),],className="card",),width=12),
                        
                        
                    ], 
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Div(children=[dcc.Graph(id = 'fig_8_2_v2',),],className="card",),width=12),
                        
                        
                    ], 
                ),
            ], fluid=True,
        ),
        #dcc.Graph(
            #id = 'fig_6_2',  
        #),
        #dcc.Graph(
            #id = 'fig_7_2',  
        #),
        #dcc.Graph(
            #id = 'fig_8_2',  
        #),
        #dcc.Graph(
            #id = 'fig_8_2_v2',  
        #),
        dbc.Container(
            [
                dbc.Row(
            [
                dbc.Col(html.Div(children=[dcc.Dropdown(
                        options=sorted(servicios_semujeres['MUNICIPIO CENTRO'].unique()),
                        value='MERIDA',
                        clearable=False,
                        id='centro'
                        ),]),width=6),
                    ], 
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Div(children=[dcc.Graph(id = 'fig_12_2',),],className="card",),width=6),
                        dbc.Col(html.Div(children=[dcc.Graph(id = 'fig_13_2',),],className="card",),width=6),
                        
                    ], 
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Div(children=[dcc.Graph(id = 'fig_10_2',),],className="card",),width=12),
                        
                    ], 
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Div(children=[dcc.Graph(id = 'fig_9_2',),],className="card",),width=12),
                        
                    ], 
                ),
            ], fluid=True,
        ),
        ],className='wrapper'),
        #dcc.Dropdown(
        #options=sorted(servicios_semujeres['MUNICIPIO CENTRO'].unique()),
        #value='MERIDA',
        #clearable=False,
        #id='centro'
        #),
        #dcc.Graph(
            #id = 'fig_12_2',  
        #),
        #dcc.Graph(
            #id = 'fig_13_2',  
        #),
        #dcc.Graph(
            #id = 'fig_10_2',  
        #),
        #dcc.Graph(
            #id = 'fig_9_2',  
        #),
        html.Br(),
        dcc.Link('Go to Page 1', href='/page-1'),
        html.Br(),
        dcc.Link('Go back to home', href='/'),
    ]
)

@callback(
    [   Output('fig_1_2', "figure"),
        Output('fig_2_2', "figure"),
        Output('fig_3_2', "figure"),
        Output('fig_4_2', "figure"),
        #Output('fig_5_2', "figure"),
        Output('fig_6_2', "figure"),
        Output('fig_7_2', "figure"),
        Output('fig_8_2', "figure"),
        Output('fig_8_2_v2', "figure"),
        Output('fig_9_2', "figure"),
        Output('fig_10_2', "figure"),
        Output('fig_11_2', "figure"),
        Output('fig_12_2', "figure"),
        Output('fig_13_2', "figure"),
        Output('fig_14_2', "figure"),
        #Output('rango_semanas', 'children')
        
    ],
    [
        #Input("date-range-2", "start_date"),
        #Input("date-range-2", "end_date"),
        #Input("year-2", "value"),
        Input("centro", "value"),
        #Input("semana", "value"),
        #Input("my-range-slider", "value")
        Input('date-range-2', 'start_date'),
        Input('date-range-2', 'end_date')
    ]
)
def update_charts_2(centro,start_date, end_date):

    """if semana != None :
        mask = (
            #& (victimas['municipiohechos'].isin(municipio) ==True)
            #(servicios_semujeres.fecha_captura >= start_date)
            #& (servicios_semujeres.fecha_captura <= end_date)
            (servicios_semujeres.año_captura==year)
            & (servicios_semujeres.semana==semana)
        )
    if slider[0]==slider[1]:
        mask = (
            #& (victimas['municipiohechos'].isin(municipio) ==True)
            #(servicios_semujeres.fecha_captura >= start_date)
            #& (servicios_semujeres.fecha_captura <= end_date)
            (servicios_semujeres.año_captura==year)
            & (servicios_semujeres.semana==slider[0])
        )
        elif slider[0] == 0 :
        mask = (
            #& (victimas['municipiohechos'].isin(municipio) ==True)
            #(servicios_semujeres.fecha_captura >= start_date)
            #& (servicios_semujeres.fecha_captura <= end_date)
            (servicios_semujeres.año_captura==year)
            & (servicios_semujeres.semana>=slider[0])
            & (servicios_semujeres.semana<=slider[1])
        )
    #elif slider[0] > 0:
    else:
        mask = (
            #& (victimas['municipiohechos'].isin(municipio) ==True)
            #(servicios_semujeres.fecha_captura >= start_date)
            #& (servicios_semujeres.fecha_captura <= end_date)
            (servicios_semujeres.año_captura==year)
            & (servicios_semujeres.semana>slider[0])
            & (servicios_semujeres.semana<=slider[1])
        )"""

    mask = (
            (servicios_semujeres.fecha_captura >= start_date)
            & (servicios_semujeres.fecha_captura <= end_date)
            #& (servicios_semujeres.año_captura==year)
        )
    
    filtered_data = servicios_semujeres.loc[mask, :]

    meses_dic = {1:'Enero',2:'Febrero',3:'Marzo',4:'Abril',5:'Mayo',6:'Junio',7:'Julio',8:'Agosto',9:'Septiembre',10:'Octubre',11:'Noviembre',12:'Diciembre'}

    #semana_inicio = str(filtered_data['inicio_semana'].min())
    #semana_fin = str(filtered_data['fin_semana'].max())
    #seleccion = 'Ha seleccionado la/s semana/s del lunes {} al domingo {}'.format(semana_inicio[:10], semana_fin[:10])

    # Fig tabla de recuentos generales
    servicios_semujeres_2022 = filtered_data
    slider = [date.fromisoformat(start_date).strftime('%d/%m/%Y'), date.fromisoformat(end_date).strftime('%d/%m/%Y')]

    servicios_tabla = filtered_data.sort_values(by=['fecha_captura','fk_euv','fk_caso'])

    servicios_tabla = servicios_tabla.replace({"mes_captura": meses_dic})

    months = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", 
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    servicios_tabla['mes_captura'] = pd.Categorical(servicios_tabla['mes_captura'], categories=months, ordered=True)

    mujeres_nuevas = pd.DataFrame(servicios_tabla.drop_duplicates(subset=['fk_euv','mes_captura','año'], keep='first').groupby(by=['mes_captura','año','seguimiento']).fk_euv.count()).rename(columns={'fk_euv':'Mujeres nuevas'}).unstack()
    mujeres_nuevas.columns = mujeres_nuevas.columns.droplevel()
    if 'Seguimiento' not in mujeres_nuevas.columns:
        mujeres_nuevas['Seguimiento']=0
    if 'Primera Vez' not in mujeres_nuevas.columns:
        mujeres_nuevas['Primera Vez']=0
    mujeres_nuevas['Total de mujeres únicas atendidas'] = mujeres_nuevas['Primera Vez'] + mujeres_nuevas['Seguimiento']
    mujeres_nuevas = mujeres_nuevas[['Total de mujeres únicas atendidas','Primera Vez','Seguimiento']]
    servicios_totales = pd.DataFrame(servicios_tabla.groupby(by=['mes_captura','año']).fk_euv.count()).rename(columns={'fk_euv':'Servicios capturados'}).sort_values(by=['año','mes_captura'])
    tabla_final = servicios_totales.merge(mujeres_nuevas, left_index=True, right_index=True)
    tabla_final.loc['<b>Total<b>']= tabla_final.sum()
    tabla_final = tabla_final.reset_index().rename(columns={'index':'Mes de captura'})
    #tabla_final = tabla_final.replace({"Mes de captura": meses_dic})
    fig_11 = go.Figure(data=[go.Table(
        header=dict(values=list(tabla_final.columns),
                    fill_color='rgb(196, 166, 230)',
                    align='left'),
        cells=dict(values=[tabla_final['Mes de captura'],tabla_final['Servicios capturados'],tabla_final['Total de mujeres únicas atendidas'],tabla_final['Primera Vez'],tabla_final['Seguimiento']],
                fill_color='lavender',
                align='left'))
    ])
    fig_11.update_layout(
            title="Tabla 1. Servicios capturados y mujeres atendidas del {} al {}".format(slider[0],slider[1]),
        )

    

    
    # Fig 5
    indicadores_caev = filtered_data.copy()
    """indicadores_caev = filtered_data.copy()
    indicador_3 = filtered_data.copy()
    mujeres_concluidas = indicador_3[indicador_3.estatus=='Concluido'].fk_euv.unique()
    mujeres_concluidas  = indicador_3[indicador_3.fk_euv.isin(mujeres_concluidas)]
    mujeres_unicas = mujeres_concluidas.fk_euv.unique()
    df = pd.DataFrame(columns=['fk_euv','fk_caso','dias','semanas','numeroservicios'])
    for mujer in mujeres_unicas:
        testito = mujeres_concluidas[mujeres_concluidas.fk_euv == mujer]

        casos_concluidos = testito[testito.estatus=='Concluido'].fk_caso.values

        servicios_concluidos = testito[testito.estatus=='Concluido'].serviciodetalle.values

        for caso in casos_concluidos:
            for servicio in servicios_concluidos:
                try:
                    test = mujeres_concluidas[(mujeres_concluidas.fk_euv == mujer)&(mujeres_concluidas.fk_caso == caso)&(mujeres_concluidas.serviciodetalle == servicio)].iloc[[0, -1]]
                    dias = test.fecha_captura.diff().tail(1)
                    semanas =  int(round(dias / np.timedelta64(1, 'W')))
                    servicioss = test.numeroservicios.tail(1)
                    test_df = pd.DataFrame({'fk_euv':mujer, 'fk_caso':caso,'dias':dias,'semanas': semanas,'numeroservicios':servicioss,'serviciodetalle':servicio})
                    df = pd.concat([df, test_df], ignore_index=True)
                except:
                    print(mujer,caso,servicio)
                    continue
    df['semanas'] = pd.to_numeric(df['semanas'])

    fig_5 = px.histogram(df, x = 'semanas',
                title='Gráfica 5. Tiempo promedio transcurrido entre el primer contacto con la usuaria y la conclusión del proceso.',
                #color_discrete_sequence = px.colors.qualitative.Prism,
                #text = 'count',
                #color = 'USUSERVICIO',
                #color = profesionistas.index,
                color_discrete_sequence = [px.colors.qualitative.Prism[1]],
                #color_discrete_sequence= px.colors.sequential.Plasma,
                labels = {'semanas': 'Semanas transcurridas', 'count':'Frecuencia'},
                nbins = 10
                )
    fig_5.update_layout(showlegend = False, yaxis_title="Frecuencia",xaxis_title="Semanas transcurridas", height=500)
    #fig.update_layout(title_y='Número de profesionistas')

    fig_5.update_layout(
        xaxis = dict(
            tickmode = 'linear',
            tick0 = 0,
            dtick = 20
        )
    )

    fig_5.update_layout(shapes=[
        # adds line at y=5
        dict(
        type= 'line',
        xref= 'x', x0= round(df.semanas.mean()), x1= round(df.semanas.mean()),
        yref= 'paper', y0= 0, y1= 1,
        line=dict(color='white', width=1, dash='dash'),
        )], height=500)

    fig_5.add_annotation(x=round(df.semanas.mean())+6, y=200,
                text="Promedio: {}".format(round(df.semanas.mean())),
                showarrow=False,
                yshift=10,
                font=dict(color='white'))"""

    # Fig 4
    indicador_servicios = filtered_data.groupby(['serviciodetalle','estatus']).count().numeroservicios.reset_index()

    indicador_servicios = indicador_servicios[indicador_servicios.estatus!='En proceso']
    
    indicador_servicios = indicador_servicios[indicador_servicios.serviciodetalle.isin(['Canalización','Psicológico','Legales y/o jurídicos','Trabajo Social'])]

    total_servicios = indicador_servicios.groupby('serviciodetalle').sum()

    porcentajes = indicador_servicios.merge(total_servicios, left_on='serviciodetalle', right_on='serviciodetalle')
    porcentajes['porcentajes'] = round(100*porcentajes['numeroservicios_x']/porcentajes['numeroservicios_y'],1)

    total_concluidos = porcentajes[porcentajes.estatus=='Concluido'].numeroservicios_x.sum()
    porcentajes_total = round(porcentajes[porcentajes.estatus=='Concluido'].numeroservicios_x.sum()/porcentajes[porcentajes.estatus=='Concluido'].numeroservicios_y.sum()*100,2)

    fig_4 = px.bar(porcentajes, x="serviciodetalle", y="porcentajes", color="estatus", 
    title="Gráfica 4. Porcentaje de procesos concluidos por área de servicio<br><b>Total de servicios concluidos:</b> {}<br><b>Porcentaje de servicios concluidos:</b> {}%".format(total_concluidos, porcentajes_total),
    hover_name='serviciodetalle',
    text= "numeroservicios_x",
    hover_data={"numeroservicios_x":True,'serviciodetalle':False},
    labels = {'serviciodetalle': 'Área de servicio', "porcentajes":'Porcentaje', 'estatus':'Estatus','numeroservicios_x':'Número de servicios'},
    #color_discrete_sequence=['rgb(95, 70, 144)', 'rgb(29, 105, 150)'],
    color_discrete_sequence=['rgb(118, 78, 159)', 'rgb(190, 186, 218)'],
    #color_discrete_sequence=['#750D86', '#FBE426'],
    #color_discrete_sequence= px.colors.sequential.Plasma,
    )
    fig_4.update_traces(textposition='outside',cliponaxis=False)
    fig_4.update_layout(height=500)
    fig_4.update_layout(yaxis_ticksuffix = '%')

    # Fig 1
    try:
        indicadores_caev = filtered_data.copy()
        indicadores_caev=indicadores_caev[(indicadores_caev.USUSERVICIO.str[:4] == 'CYUC')]
        grup = pd.DataFrame(indicadores_caev.groupby(['USUSERVICIO','fecha_captura']).numeroservicios.count()).reset_index()
        grupito = pd.DataFrame(grup.groupby(['USUSERVICIO',pd.Grouper(freq='W', key='fecha_captura')]).sum()).reset_index()
        grupito['semana'] = grupito.fecha_captura.dt.strftime('%W').astype(int)
        grupito = grupito.sort_values(by='fecha_captura')
        start = grupito.drop_duplicates(subset=['USUSERVICIO'], keep='first')
        end = grupito.drop_duplicates(subset=['USUSERVICIO'], keep='last')
        grupito=pd.DataFrame(grupito.groupby(by='USUSERVICIO').numeroservicios.sum()).round().reset_index()
        grupito=grupito.merge(start[['USUSERVICIO','fecha_captura']], on='USUSERVICIO', how='left').merge(end[['USUSERVICIO','fecha_captura']], on='USUSERVICIO', how='left')
        grupito['dif_capturas'] = ((grupito['fecha_captura_y'] - grupito['fecha_captura_x']).dt.days/7).round()+1
        grupito['promedio']=(grupito.numeroservicios/grupito['dif_capturas']).round()
        grupito=grupito.sort_values(by='promedio', ascending=False)
        media_servicios_semanales_round = round(grupito.promedio.mean(),)
        grupito=grupito.merge(centros_no_duplicados[['CUENTA','NOMBRE']], left_on='USUSERVICIO', right_on='CUENTA', how='left').drop('CUENTA', axis=1)
        grupito['NOMBRE']=grupito.NOMBRE.str.strip()
        grupito.loc[grupito.NOMBRE.isna(),'NOMBRE']='DESCONOCIDO'
        grupito['fecha_captura_x'] = grupito['fecha_captura_x'].dt.date
        grupito['fecha_captura_y'] = grupito['fecha_captura_y'].dt.date

        fig_1 = px.bar(grupito, 
                        x = grupito.USUSERVICIO, 
                        y = 'promedio', 
                        color_continuous_scale='Plasma',
                        hover_name='USUSERVICIO',
                        hover_data={'USUSERVICIO':False, 'fecha_captura_x': True, 'fecha_captura_y': True, 'NOMBRE': True, 'dif_capturas':True, 'promedio':True,'numeroservicios':False},
                        color = 'dif_capturas',
                        labels = {'promedio':'Promedio de servicios semanales','USUSERVICIO': 'Clave del profesionista', "fecha_captura_x":'Semana inicio captura', 'fecha_captura_y': 'Semana última captura','dif_capturas':'Semanas capturando'},
                        title= 'Gráfica 1. Promedio de servicios semanales otorgados por profesionistas de la SEMUJERES')
        fig_1.update_layout(showlegend = False)

        #print("plotly express hovertemplate:", fig.data[0].hovertemplate)
        fig_1.update_traces(hovertemplate='<b>%{hovertext}</b><br><b>%{customdata[2]}</b><br><br>Promedio de servicios semanales=%{y}<br>Semana inicio captura=%{customdata[0]}<br>Semana última captura=%{customdata[1]}<br>Semanas capturando=%{marker.color}<extra></extra>')

        fig_1.update_layout(legend_title= False, height=500)
        #fig.update(layout_coloraxis_showscale=False)
        fig_1.update_coloraxes(colorbar_title=None)

        fig_1.update_layout(shapes=[
            # adds line at y=5
            dict(
            type= 'line',
            xref= 'paper', x0= 0, x1= 1,
            yref= 'y', y0= media_servicios_semanales_round, y1= media_servicios_semanales_round,
            line=dict(color='black', width=1, dash='dash'),
            )])

        fig_1.add_annotation(x=45, y=media_servicios_semanales_round,
                    text="Promedio de servicios semanales: {}".format(media_servicios_semanales_round),
                    showarrow=False,
                    yshift=10)

        fig_1.update_xaxes(
                tickangle = 70)
                
        fig_1.update_layout(
            coloraxis_colorbar={
                'title':'Semanas<br>Capturando'})

        # fIG 2 
        media_servicios_semanales_round = round(grupito.numeroservicios.mean(),)
        grupito = grupito.sort_values(by='numeroservicios',ascending=False)
        fig_2 = px.bar(grupito, 
                        x = grupito.USUSERVICIO, 
                        y = 'numeroservicios', 
                        color_continuous_scale='Plasma',
                        hover_name='USUSERVICIO',
                        hover_data={'USUSERVICIO':False, 'fecha_captura_x': True, 'fecha_captura_y': True, 'NOMBRE': True, 'dif_capturas':True, 'promedio':True,'numeroservicios':True},
                        color = 'dif_capturas',
                        labels = {'numeroservicios':'Servicios totales','promedio':'Promedio de servicios semanales','USUSERVICIO': 'Clave del profesionista', "fecha_captura_x":'Semana inicio captura', 'fecha_captura_y': 'Semana última captura','dif_capturas':'Semanas capturando'},
                        title= 'Gráfica 2. Servicios totales otorgados por profesionistas de la SEMUJERES')
        fig_2.update_layout(showlegend = False)

        #print("plotly express hovertemplate:", fig.data[0].hovertemplate)
        fig_2.update_traces(hovertemplate=' <b>%{hovertext}</b><br><b>%{customdata[2]}</b><br><br>Servicios totales=%{y}<br>Semana inicio captura=%{customdata[0]}<br>Semana última captura=%{customdata[1]}<br>Semanas capturando=%{marker.color}<br>Promedio de servicios semanales=%{customdata[4]}<extra></extra>')

        fig_2.update_layout(legend_title= False, height=500)
        #fig.update(layout_coloraxis_showscale=False)
        fig_2.update_coloraxes(colorbar_title=None)

        fig_2.update_layout(shapes=[
            # adds line at y=5
            dict(
            type= 'line',
            xref= 'paper', x0= 0, x1= 1,
            yref= 'y', y0= media_servicios_semanales_round, y1= media_servicios_semanales_round,
            line=dict(color='black', width=1, dash='dash'),
            )])

        fig_2.add_annotation(x=45, y=media_servicios_semanales_round,
                    text="Promedio de servicios: {}".format(media_servicios_semanales_round),
                    showarrow=False,
                    yshift=10)

        fig_2.update_xaxes(
                tickangle = 70)
                
        fig_2.update_layout(
            coloraxis_colorbar={
                'title':'Semanas<br>Capturando'})
    except:
        fig_1=px.bar(title='Gráfica 1. SIN SERVICIOS CAPTURADOS PARA ESTE INTERVALO')
        fig_2=px.bar(title='Gráfica 2. SIN SERVICIOS CAPTURADOS PARA ESTE INTERVALO')

    # Fig 3
    media_servicios = round(grupito.promedio.mean())
    #profesionistas.numeroservicios = profesionistas.numeroservicios.round()

    fig_3 = px.histogram(grupito, x = 'promedio',
                title='Gráfica 3. Distribución del número de Servicios Semanales otorgados por profesionista en SEMUJERES',
                #color_discrete_sequence = px.colors.qualitative.Prism,
                #text = 'count',
                #color = 'USUSERVICIO',
                #color = profesionistas.index,
                color_discrete_sequence = [px.colors.qualitative.Bold[0]],
                #color_discrete_sequence = ['#750D86'],
                #color_discrete_sequence= px.colors.sequential.Plasma,
                labels = {'promedio': 'Número de Servicios Semanales', 'count':'ola'},
                #nbins = 9,
                #range_x=[0, 44]
                )
    fig_3.update_layout(showlegend = False, yaxis_title="Frecuencia", height=500)
    #fig.update_layout(title_y='Número de profesionistas')
    fig_3.update_layout(
        xaxis = dict(
            tickmode = 'linear',
            tick0 = 0,
            dtick = 5
        )
    )
    fig_3.update_layout(shapes=[
        # adds line at y=5
        dict(
        type= 'line',
        xref= 'x', x0= media_servicios, x1= media_servicios,
        yref= 'paper', y0= 0, y1= 1,
        line=dict(color='black', width=1, dash='dash'),
        )])

    fig_3.add_annotation(x=media_servicios+1.5, y=10,
                text="Promedio: {}".format(media_servicios),
                showarrow=False,
                yshift=10,
                font=dict(color='black'))

    # Fig 6
    #final = filtered_data.merge(centros_no_duplicados, left_on='USUSERVICIO', right_on='CUENTA', how='left').drop(['CUENTA'],1)
    #final['MUNICIPIO CENTRO'] = final['MUNICIPIO CENTRO'].fillna('DESCONOCIDO')
    final = filtered_data.copy()
    centros_bar = pd.DataFrame(final['MUNICIPIO CENTRO'].value_counts()).reset_index()
    centros_bar['log'] = np.log(centros_bar['MUNICIPIO CENTRO'])
    fig_6 = px.bar(centros_bar, 
                x = 'index', 
                y = 'MUNICIPIO CENTRO', 
                hover_name='index',
                color_continuous_scale='Sunsetdark',
                hover_data={ 'index': False,'log':False},
                color = 'log',
                labels = {'MUNICIPIO CENTRO': 'Servicios totales', 'index':'Centro'},
                title= 'Gráfica 5. Servicios totales otorgados por centros de la SEMUJERES')
    fig_6.update_layout(showlegend = False, xaxis_title=None)

    #print("plotly express hovertemplate:", fig.data[0].hovertemplate)
    #fig.update_traces(hovertemplate='<b>%{hovertext}</b><br>Nombre=%{customdata[0]}<br><b>Servicios otorgados</b>=%{y}<br>Primera captura=%{customdata[1]}<br>Última captura=%{customdata[2]}<br>Semanas capturando=%{customdata[3]} semanas<extra></extra>')

    fig_6.update_layout(legend_title= False, height=500, coloraxis_showscale=False)

    # Fig 7
    
    unicas = final.drop_duplicates(subset=['fk_euv','MUNICIPIO CENTRO'])
    unicas= pd.DataFrame(unicas['MUNICIPIO CENTRO'].value_counts()).reset_index()
    unicas['log'] = np.log(unicas['MUNICIPIO CENTRO'])
    fig_7 = px.bar(unicas, 
                x = 'index', 
                y = 'MUNICIPIO CENTRO', 
                color_continuous_scale='Sunsetdark',
                hover_name='index',
                #hover_name='fk_usuario',
                #hover_data={'fk_usuario':False, 'NomCompleto':True,'Semana de primera captura': True, 'Semana de última captura':True, 'dif_capturas': True},
                #color = 'log',
                #labels = {'USUSERVICIO': 'Servicios totales', "fk_usuario":'Clave del profesionista', 'NomCompleto': 'Nombre'},
                hover_data={ 'index': False,'log':False},
                color = 'log',
                labels = {'MUNICIPIO CENTRO': 'Mujeres atendidas', 'index':'Centro'},
                title= 'Gráfica 6. Mujeres únicas atendidas por centros de la SEMUJERES')
    fig_7.update_layout(showlegend = False, xaxis_title=None)

    #print("plotly express hovertemplate:", fig.data[0].hovertemplate)
    #fig.update_traces(hovertemplate='<b>%{hovertext}</b><br>Nombre=%{customdata[0]}<br><b>Servicios otorgados</b>=%{y}<br>Primera captura=%{customdata[1]}<br>Última captura=%{customdata[2]}<br>Semanas capturando=%{customdata[3]} semanas<extra></extra>')

    fig_7.update_layout(legend_title= False, height=500, coloraxis_showscale=False)

    # Fig 8
    centros_bar = pd.DataFrame(final[['MUNICIPIO CENTRO','serviciodetalle']].value_counts()).reset_index().rename(columns={0:'count'})
    centros_bar=centros_bar.groupby(by=['MUNICIPIO CENTRO','serviciodetalle']).agg({'count': 'sum'}).groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).rename(columns={'count':'percent'}).reset_index()
    centros_bar=centros_bar.merge(pd.DataFrame(final[['MUNICIPIO CENTRO','serviciodetalle']].value_counts()).reset_index().rename(columns={0:'count'}), on=['MUNICIPIO CENTRO','serviciodetalle'], how='right')
    centros_bar['percent_dec']  = centros_bar['percent'].round(1)
    centros_bar['percent']  = centros_bar['percent'].round(1).astype(str) + '%'
    centros_bar = centros_bar[centros_bar.serviciodetalle.isin(['Legales y/o jurídicos','Psicológico','Violencia y género','Canalización','Trabajo Social'])]

    fig_8 = px.bar(centros_bar, 
                    x = 'MUNICIPIO CENTRO', 
                    y = 'count', 
                    #color_continuous_scale='Viridis',
                    color_discrete_sequence= px.colors.qualitative.Prism,
                    hover_name='serviciodetalle',
                    hover_data={'percent':True,'MUNICIPIO CENTRO': True,'serviciodetalle':False},
                    color = 'serviciodetalle',
                    #hover_data={ },
                    labels = {'MUNICIPIO CENTRO': 'Centro', 'count':'Número de servicios','percent':'Porcentaje','serviciodetalle':'Tipo de servicio'},
                    #labels = {'USUSERVICIO': 'Servicios totales', "fk_usuario":'Clave del profesionista', 'NomCompleto': 'Nombre'},
                    title= 'Gráfica 7. Tipos de servicio otorgados por centros de la SEMUJERES')
    fig_8.update_layout(showlegend = True, xaxis_title=None)

    #print("plotly express hovertemplate:", fig.data[0].hovertemplate)
    #fig.update_traces(hovertemplate='<b>%{hovertext}</b><br>Nombre=%{customdata[0]}<br><b>Servicios otorgados</b>=%{y}<br>Primera captura=%{customdata[1]}<br>Última captura=%{customdata[2]}<br>Semanas capturando=%{customdata[3]} semanas<extra></extra>')

    fig_8.update_layout(legend_title= 'Tipo de servicio', height=500, )

    # Fig 8 version valeria
    
    ola = centros_bar[['MUNICIPIO CENTRO','serviciodetalle','percent_dec']]
    ola = ola.pivot(index='MUNICIPIO CENTRO',columns='serviciodetalle',values='percent_dec')
    ola.fillna(0, inplace=True)
    fig_8_2 = px.imshow(ola, labels=dict(x='Servicio', y="Centro Violeta", color="Porcentaje"),aspect='auto',  text_auto='.2f', color_continuous_scale='Purples', title='Gráfica 7.2 Porcentaje de tipos de servicios otorgados por centros de la SEMUJERES')
    fig_8_2.update_xaxes(side="top")
    fig_8_2.update_traces(text=ola.values.astype(str),texttemplate="%{text} %")
    fig_8_2.update_layout(coloraxis_showscale=False,xaxis_title=None)
    

    # Fig 9
    violencias=final.merge(victimas[['fk_euv','pk_caso','Económica', 'Física','Patrimonial', 'Psicológica', 'Sexual', 'Feminicida']], left_on=['fk_euv','fk_caso'], right_on=['fk_euv','pk_caso'], how='left').drop_duplicates(subset=['fk_euv','pk_caso'], keep='last')
    casos = violencias[violencias['MUNICIPIO CENTRO']==centro]
    temp = casos[['Económica', 'Física','Patrimonial', 'Psicológica', 'Sexual', 'Feminicida']].sum()
    temp = pd.DataFrame(temp.sort_values(ascending=False))

    y = temp[0].values
    y_total = len(casos)
    fig_9 = px.bar(x=temp.index,
                y=temp[0],
                text= np.round(y/y_total*100,2),
                labels = {'x': 'Tipo de violencia', "y":'Número de casos', 'text':'Porcentaje'},
                color=px.colors.sequential.matter[::-1][:6][::],
                color_discrete_map="identity",
                #color_discrete_sequence= px.colors.sequential.Plasma_r,
                #color_continuous_scale='matter_r',
                #hover_name='x',
                title='Gráfica 8. Porcentaje de casos registrados en el centro de {} por tipo de violencia'.format(centro)
                )
    fig_9.update_xaxes(type='category')
    fig_9.update_layout( xaxis_title=None,  height=500)
    fig_9.update_traces(texttemplate='%{text} %')

    # Fig 10
    tabla = pd.DataFrame(final.groupby(by=['MUNICIPIO CENTRO','USUSERVICIO']).fk_euv.count())
    tabla=tabla.reset_index()
    tabla=tabla[tabla['MUNICIPIO CENTRO']==centro]
    tabla = tabla.merge(centros_no_duplicados, left_on='USUSERVICIO',right_on='CUENTA', how='left').drop(['MUNICIPIO CENTRO_y','CUENTA'], axis=1).rename(columns={'MUNICIPIO CENTRO_x':'MUNICIPIO CENTRO'})
    tabla = tabla.sort_values(by='fk_euv', ascending=False)
    tabla = tabla.rename(columns={'fk_euv':'SERVICIOS OTORGADOS','USUSERVICIO':'USUARIO','MUNICIPIO CENTRO':'CENTRO'})
    tabla.loc[tabla['CENTRO'].duplicated(), 'CENTRO'] = ''
    tabla = tabla[['CENTRO','USUARIO','NOMBRE','SERVICIOS OTORGADOS']]
    tabla.loc['<b>Total<b>'] = ['','','<b>Total<b>',tabla['SERVICIOS OTORGADOS'].sum()]
    fig_10 = go.Figure(data=[go.Table(
        header=dict(values=list(tabla.columns),
                    fill_color='rgb(196, 166, 230)',
                    align='left'),
        cells=dict(values=[tabla['CENTRO'],tabla['USUARIO'],tabla['NOMBRE'],tabla['SERVICIOS OTORGADOS']],
                fill_color='lavender',
                align='left'))
    ])
    
    fig_10.update_layout(
        title="Tabla 5. Servicios otorgados por profesionista en el centro {} del {} al {}".format(centro, slider[0],slider[1]),
    )

    # Fig tabla de recuentos generales por centro
    #servicios_semujeres_2022 = servicios_semujeres[(servicios_semujeres.año_captura==year)&(servicios_semujeres['MUNICIPIO CENTRO']==centro)]
    servicios_semujeres_2022=final[final['MUNICIPIO CENTRO']==centro].copy()
    servicios_tabla = servicios_semujeres_2022.sort_values(by='fecha_captura')

    servicios_tabla = servicios_tabla.replace({"mes_captura": meses_dic})

    months = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", 
            "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    servicios_tabla['mes_captura'] = pd.Categorical(servicios_tabla['mes_captura'], categories=months, ordered=True)

    mujeres_nuevas = pd.DataFrame(servicios_tabla.drop_duplicates(subset=['fk_euv','año'], keep='first').groupby(by=['mes_captura','año']).fk_euv.count()).rename(columns={'fk_euv':'Mujeres nuevas'})#.sort_values(by=['año','mes_captura'])
    mujeres_unicas_atendidas = pd.DataFrame(servicios_tabla.drop_duplicates(subset=['fk_euv','mes_captura','año'], keep='first').groupby(by=['mes_captura','año']).fk_euv.count()).rename(columns={'fk_euv':'Mujeres únicas atendidas'})#.sort_values(by=['año','mes_captura'])
    servicios_totales = pd.DataFrame(servicios_tabla.groupby(by=['mes_captura','año']).fk_euv.count()).rename(columns={'fk_euv':'Servicios capturados'}).sort_values(by=['año','mes_captura'])
    tabla_final = servicios_totales.merge(mujeres_nuevas, left_index=True, right_index=True).merge(mujeres_unicas_atendidas, left_index=True, right_index=True)
    tabla_final['Mujeres en seguimiento'] = tabla_final['Mujeres únicas atendidas']-tabla_final['Mujeres nuevas']
    tabla_final = tabla_final[['Servicios capturados','Mujeres únicas atendidas','Mujeres nuevas','Mujeres en seguimiento']]
    tabla_final.loc['<b>Total<b>']= tabla_final.sum()
    tabla_final = tabla_final.reset_index().rename(columns={'index':'Mes de captura'})
    #tabla_final = tabla_final.replace({"Mes de captura": meses_dic})
    fig_12 = go.Figure(data=[go.Table(
        header=dict(values=list(tabla_final.columns),
                    fill_color='rgb(196, 166, 230)',
                    align='left'),
        cells=dict(values=[tabla_final['Mes de captura'],tabla_final['Servicios capturados'],tabla_final['Mujeres únicas atendidas'],tabla_final['Mujeres nuevas'],tabla_final['Mujeres en seguimiento']],
                fill_color='lavender',
                align='left'))
    ])
    fig_12.update_layout(
            title="Tabla 3. Servicios capturados y mujeres atendidas por mes en el centro {}<br>de {} a {}".format(centro,slider[0],slider[1]),
            height=400,
        )
    
    # Fig de expediente no envidados 
    tabla=status_tabla[status_tabla['MUNICIPIO CENTRO']==centro]
    tabla = tabla.rename(columns={'año_recepcion':'Año de recepción','MUNICIPIO CENTRO':'Centro'})
    tabla.loc[tabla['Centro'].duplicated(), 'Centro'] = ''
    tabla = tabla[['Centro','Año de recepción','Expedientes no enviados']]
    tabla.loc['<b>Total<b>']= ['','<b>Total<b>',tabla['Expedientes no enviados'].sum()]
    fig_13 = go.Figure(data=[go.Table(
        header=dict(values=list(tabla.columns),
                    fill_color='rgb(196, 166, 230)',
                    align='left'),
        cells=dict(values=[tabla['Centro'],tabla['Año de recepción'],tabla['Expedientes no enviados']],
                fill_color='lavender',
                align='left'))
    ])
    fig_13.update_layout(
            title="Tabla 4. Expedientes no enviados en el centro {} por año".format(centro),
            height=400,
        )

    # Figura 14 servicios por tipo
    servicios_semujeres_2022 = filtered_data
    servicios_tabla = servicios_semujeres_2022.sort_values(by='fecha_captura')
    servicios_tabla = servicios_tabla.replace({"mes_captura": meses_dic})

    servicios_tabla['mes_captura'] = pd.Categorical(servicios_tabla['mes_captura'], categories=months, ordered=True)

    servicios_totales = pd.DataFrame(servicios_tabla.groupby(by=['mes_captura','año']).fk_euv.count()).rename(columns={'fk_euv':'Total'}).sort_values(by=['año','mes_captura'])
    servicios_tabla = servicios_tabla.groupby(by=['mes_captura','año','serviciodetalle']).fk_euv.count().unstack('serviciodetalle')
    tabla_final = servicios_totales.merge(servicios_tabla, left_index=True, right_index=True)
    tabla_final.loc['<b>Total<b>']= tabla_final.sum()
    tabla_final=tabla_final.fillna(0)
    tabla_final = tabla_final.reset_index().rename(columns={'index':'Mes de captura'})
    fig_14 = go.Figure(data=[go.Table(
        header=dict(values=list(tabla_final.columns),
                    fill_color='rgb(196, 166, 230)',
                    align='left'),
        cells=dict(values=[tabla_final[col] for col in tabla_final.columns],
                fill_color='lavender',
                align='left'))
    ])
    fig_14.update_layout(
            title="Tabla 2. Total de servicios capturados por tipo del {} al {}".format(slider[0],slider[1]),
            height=350, 
        )

    return fig_1,fig_2,fig_3,fig_4,fig_6, fig_7, fig_8, fig_8_2, fig_9, fig_10, fig_11, fig_12, fig_13, fig_14

# Update the index
@callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        #return page_1_layout
        return None
    elif pathname == '/page-2':
        return page_2_layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here


if __name__ == "__main__":
    app.run_server()


# In[ ]:




