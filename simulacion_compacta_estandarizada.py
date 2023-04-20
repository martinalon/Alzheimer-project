import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from scipy import stats as stats
from scipy.stats import rv_continuous, rv_histogram, norm, beta, multivariate_normal

import statsmodels.distributions.empirical_distribution as edf
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from scipy.stats import t
import statsmodels.api as sm



#######################################################    simulacion de las uniformes correlacionadas 

def mis_uniformes(datos, n):
    Corr = datos.corr()                                 # Matriz de correlacion
    m0 = np.zeros(3)                                    # Vector de medias en la normal centrada
    mv_norm = multivariate_normal(mean=m0, cov=Corr)    #  cov = matrix de correlacion
    rand_Nmv = mv_norm.rvs(n)                           #  simulacion de  variables aleatorias de la normal multivariada  
    rand_U =  (rand_Nmv)                         #  simulacion de las variables aleatorias uniformes ya correlacionadas
    rand_U_df = {'mc': rand_U[:,0], 'edad':rand_U[:,1] , 'escol':rand_U[:,2] }
    rand_U_df = pd.DataFrame(rand_U_df)
    return(rand_U_df)

#########################################3
###########################################
# Distribucion empirica inversa 

def inversas_cdf(datos, a, a_max, b, c):
    # definicion de la media y desviacion estandar de cada columna en el dataframe
    a_media = datos[a].mean()
    a_std = datos[a].std()
    #a_max = datos[a].max()  # el maximo de esta columna depende a los maximos considerados en la prueva
    
    b_media = datos[b].mean()
    b_std = datos[b].std()
    b_max = 55                # maximo de edad para pasientes preclinicos 
    
    #c_media = datos[c].mean()
    c_std = datos[c].std()
    c_max =  24               # maxima escolaridad tomada en cuenta 
    
    datos[a] = (datos[a]- a_media)/a_std   # estandarizacion de los resultados de la prueba
    datos[b] = (datos[b] - b_media)/ b_std # estandarizacion de los resultados de la edad
    datos[c] = (datos[c] - 9.9)/c_std      # # estandarizacion de los resultados de la escolaridad con media regional
    
    rand_U_df0 = mis_uniformes(datos, 15000)
    percentiles = pd.DataFrame()
    # a continuacion se toman los percentiles, es decir los nuevos datos antes de limpiarlos 
    percentiles[a] = norm.ppf(rand_U_df0["mc"])   
    percentiles[b] = norm.ppf(rand_U_df0["edad"])
    percentiles[c] = norm.ppf(rand_U_df0["escol"])
    # A continuacion se limpian los nuevos datos 
    nuevos_datos = percentiles[(percentiles[a] <= (a_max-a_media)/a_std) & (percentiles[a] >= (-a_media)/a_std)] # resultados no negativos y menores o iguales a 30
    nuevos_datos = percentiles[(percentiles[b] <= (b_max-b_media)/b_std) & (percentiles[b] >= (18 - b_media)/b_std)] # edad mayor a 18 años y menor a 55
    nuevos_datos = percentiles[(percentiles[c] <= (c_max-9.9)/c_std) & (percentiles[c] >= (-9.9)/c_std)]   #escolaridad no negativa y menor a 24 años
    
    return(nuevos_datos.head(10000))

##########################################
##########################################
##########################################  en esta funcion se hace la regresion lineal y se regresa el resumen de los resultados para los modelos con y sin intercepto
def resumen_RL(Y, X_L, X_C):
    # estos son los parametros de la regresion lineal con intercepto
    parametros_L = pd.DataFrame(columns=['b inter', 'b edad', 'b escol', 't0 inter', 't0 edad', 't0 escol', 'p inter', 'p edad', 'p escol'])
    model_L = sm.OLS(Y,X_L)
    results_L = model_L.fit()
    resume_names = ['coef', 't', 'P>|t|']

    columns_L = ['b inter', 'b edad', 'b escol', 't0 inter', 't0 edad',  't0 escol', 'p inter', 'p edad', 'p escol']
    param_names_L = ['unos','EDAD', 'ESCOLARIDAD' ]

    k = 0
    for i in resume_names:
        for j in  param_names_L:
            parametros_L[columns_L[k]] = results_L.t_test(j).summary_frame()[i]
            k= k +1
    
    # estos son los parametros de la regresion cuadratica con intercepto
    parametros_C = pd.DataFrame(columns=['b inter', 'b edad', 'b edad_cuad', 'b escol', 't0 inter', 't0 edad', 't0 edad_cuad', 't0 escol', 'p inter', 'p edad', 'p edad cuad', 'p escol'])
    model_C = sm.OLS(Y,X_C)
    results_C = model_C.fit()
    resume_names = ['coef', 't', 'P>|t|']

    columns_C = ['b inter', 'b edad', 'b edad_cuad', 'b escol', 't0 inter', 't0 edad', 't0 edad_cuad', 't0 escol', 'p inter', 'p edad', 'p edad cuad', 'p escol']
    param_names_C = ['unos','EDAD', 'edad_cuad', 'ESCOLARIDAD' ]

    k = 0
    for i in resume_names:
        for j in  param_names_C:
            parametros_C[columns_C[k]] = results_C.t_test(j).summary_frame()[i]
            k= k +1


    return(parametros_L, parametros_C)

#######################################
####################################### modelos de regresion 
def modelo_C(parametros, x1, x2):
    b0 = parametros[0]
    b1 = parametros[1]
    b2 = parametros[2]
    b3 = parametros[3]
    y = b0 + b1*x1 + b2*(x1**2) + b3*x2
    return(y)

def modelo_L(parametros, x1, x2):
    b0 = parametros[0]
    b1 = parametros[1]
    b2 = parametros[2]
    y = b0 + b1*x1 + b2*x2
    return(y)
####################################3
#######################################
#######################################  Funcion para recuperar las suma de los errores al cuadrado r


def sum_dif_cuad(datos_gp, datos_gnp, name, parametrosgp_L_media, parametrosgnp_L_media, parametrosgp_C_media, parametrosgnp_C_media): 
    y_gp = []
    y_hat_gp_L = []
    y_hat_gp_C = []
    for i in range(0, len(datos_gp)):
        y_gp.append(datos_gp.iloc[i,0])
        y_hat_gp_L.append(modelo_L(parametrosgp_L_media, datos_gp.iloc[i,1], datos_gp.iloc[i,2]))
        y_hat_gp_C.append(modelo_C(parametrosgp_C_media, datos_gp.iloc[i,1], datos_gp.iloc[i,2]))

    y_gnp = []
    y_hat_gnp_L = []
    y_hat_gnp_C = []
    for i in range(0, len(datos_gnp)):
        y_gnp.append(datos_gnp.iloc[i,0])
        y_hat_gnp_L.append(modelo_L(parametrosgnp_L_media, datos_gnp.iloc[i,1], datos_gnp.iloc[i,2]))
        y_hat_gnp_C.append(modelo_C(parametrosgnp_C_media, datos_gnp.iloc[i,1], datos_gnp.iloc[i,2]))

    sum_dif_cuad = {
        'gp_L':(np.asarray(y_gp) - np.asarray(y_hat_gp_L)) @ (np.asarray(y_gp) - np.asarray(y_hat_gp_L)),
        'gp_C':(np.asarray(y_gp) - np.asarray(y_hat_gp_C)) @ (np.asarray(y_gp) - np.asarray(y_hat_gp_C)),
        'gnp_L': (np.asarray(y_gnp) - np.asarray(y_hat_gnp_L)) @ (np.asarray(y_gnp) - np.asarray(y_hat_gnp_L)),
        'gnp_C': (np.asarray(y_gnp) - np.asarray(y_hat_gnp_C)) @ (np.asarray(y_gnp) - np.asarray(y_hat_gnp_C))
        } 
    sum_dif_cuad = pd.DataFrame(sum_dif_cuad, index= [name])
    return(sum_dif_cuad)

###########################################
###########################################
###########################################   funcion que da el resumen de las regreciones se obtinen los resultados de las regresiones 

def parametros_info_df(datos_gp, datos_gnp, name_L, name_C, a, f_normalizacion, b, c):
    parametrosgp_L = pd.DataFrame(columns=['b inter', 'b edad', 'b escol', 't0 inter', 't0 edad', 't0 escol', 'p inter', 'p edad', 'p escol'])
    parametrosgp_C = pd.DataFrame(columns=['b inter', 'b edad', 'b edad_cuad', 'b escol', 't0 inter', 't0 edad', 't0 edad_cuad', 't0 escol', 'p inter', 'p edad', 'p edad cuad', 'p escol'])
    parametrosgnp_L = pd.DataFrame(columns=['b inter', 'b edad', 'b escol', 't0 inter', 't0 edad', 't0 escol', 'p inter', 'p edad', 'p escol'])
    parametrosgnp_C = pd.DataFrame(columns=['b inter', 'b edad', 'b edad_cuad', 'b escol', 't0 inter', 't0 edad', 't0 edad_cuad', 't0 escol', 'p inter', 'p edad', 'p edad cuad', 'p escol'])

    for i in range(0,10):
        datos_n_gp = inversas_cdf(datos_gp, a, f_normalizacion, b, c)                                   #Datos simulados
        datos_n_gp['unos'] = np.ones(datos_n_gp.shape[0])
        datos_n_gp['edad_cuad'] = datos_n_gp[b].mul(datos_n_gp[b])     # agrgando el cuadrado de las edades
        Xgp_L =  datos_n_gp[['unos',b, c ]]
        Xgp_C =  datos_n_gp[['unos',b, 'edad_cuad', c]]
        Ygp = datos_n_gp[a]
        resumen_gp = resumen_RL(Ygp, Xgp_L, Xgp_C)
        resumen_gp_L = resumen_gp[0]
        resumen_gp_C = resumen_gp[1]
        parametrosgp_L = pd.concat([parametrosgp_L, resumen_gp_L], axis=0)
        parametrosgp_C = pd.concat([parametrosgp_C, resumen_gp_C], axis=0)

        datos_n_gnp = inversas_cdf(datos_gnp, a, f_normalizacion, b, c)                                   #Datos simulados
        datos_n_gnp['unos'] = np.ones(datos_n_gnp.shape[0])
        datos_n_gnp['edad_cuad'] = datos_n_gnp[b].mul(datos_n_gnp[b])     # agrgando el cuadrado de las edades
        Xgnp_L =  datos_n_gnp[['unos',b, c ]]
        Xgnp_C =  datos_n_gnp[['unos',b, 'edad_cuad', c ]]
        Ygnp = datos_n_gnp[a]
        resumen_gnp = resumen_RL(Ygnp, Xgnp_L, Xgnp_C)
        resumen_gnp_L = resumen_gnp[0]
        resumen_gnp_C = resumen_gnp[1]
        parametrosgnp_L = pd.concat([parametrosgnp_L, resumen_gnp_L], axis=0)
        parametrosgnp_C = pd.concat([parametrosgnp_C, resumen_gnp_C], axis=0)
        print(i)
    
    parametrosgnp_L_clean = parametrosgnp_L[(parametrosgnp_L['p escol'] < 0.05) &   (parametrosgnp_L['p edad'] < 0.05) &  (parametrosgnp_L['p inter'] < 0.05)]
    parametrosgp_L_clean = parametrosgp_L[(parametrosgp_L['p escol'] < 0.05) &   (parametrosgp_L['p edad'] < 0.05) &  (parametrosgp_L ['p inter'] < 0.05)]
    parametrosgnp_C_clean = parametrosgnp_C[(parametrosgnp_C['p escol'] < 0.05) & (parametrosgnp_C['p edad cuad'] < 0.05) &  (parametrosgnp_C['p edad'] < 0.05) &  (parametrosgnp_C ['p inter'] < 0.05)]
    parametrosgp_C_clean = parametrosgp_C[(parametrosgp_C['p escol'] < 0.05) & (parametrosgp_C['p edad cuad'] < 0.05) &  (parametrosgp_C['p edad'] < 0.05) &  (parametrosgp_C ['p inter'] < 0.05)]
    
    ind = ['count_gnp', 'mean_gnp', 'std_gnp', 'min_gnp', '25%_gnp', '50%_gnp', '75%_gnp', 'max_gnp',
           'count_gp', 'mean_gp', 'std_gp', 'min_gp', '25%_gp', '50%_gp', '75%_gp', 'max_gp']

    resumen_L = pd.concat([parametrosgnp_L_clean.describe(), parametrosgp_L_clean.describe()], axis=0)
    resumen_L['new_index'] = ind
    resumen_L.to_csv(name_L)
    resumen_C = pd.concat([parametrosgnp_C_clean.describe(), parametrosgp_C_clean.describe()], axis=0)
    resumen_C['new_index'] = ind
    resumen_C.to_csv(name_C)

    parametrosgp_L_media = parametrosgp_L_clean[['b inter', 'b edad', 'b escol']].mean()
    parametrosgnp_L_media = parametrosgnp_L_clean[['b inter', 'b edad', 'b escol']].mean()
    parametrosgp_C_media = parametrosgp_C_clean[['b inter', 'b edad', 'b edad_cuad', 'b escol']].mean()
    parametrosgnp_C_media = parametrosgnp_C_clean[['b inter', 'b edad', 'b edad_cuad', 'b escol']].mean()

    return(parametrosgp_L_media, parametrosgnp_L_media, parametrosgp_C_media, parametrosgnp_C_media)
########################################################
########################################################
########################################################
########################################################

def plt_3d(name, name_3dplot, parametrosgp_L_media, parametrosgnp_L_media, parametrosgp_C_media, parametrosgnp_C_media ):
    edad = np.linspace(-2, 3, 100) 
    escolaridad = np.linspace(-3, 3, 100) 
    EDAD, ESCOLARIDAD = np.meshgrid(edad, escolaridad)

    rendimientogp_L_promedio = modelo_L(parametrosgp_L_media  , EDAD, ESCOLARIDAD )
    rendimientognp_L_promedio = modelo_L(parametrosgnp_L_media , EDAD, ESCOLARIDAD )
    rendimientogp_C_promedio = modelo_C(parametrosgp_C_media  , EDAD, ESCOLARIDAD )
    rendimientognp_C_promedio = modelo_C(parametrosgnp_C_media , EDAD, ESCOLARIDAD )

    min_value = min([rendimientogp_L_promedio.min(), rendimientognp_L_promedio.min(), rendimientogp_C_promedio.min(), rendimientognp_C_promedio.min()])
    max_value = max([rendimientogp_L_promedio.max(), rendimientognp_L_promedio.max(), rendimientogp_C_promedio.max(), rendimientognp_C_promedio.max()])
    
    font = {'size': 4}   
    plt.rc('font', **font)
    fig = plt.figure()
    ax =  plt.axes(projection='3d')
    ax.plot_surface(EDAD, ESCOLARIDAD, rendimientogp_L_promedio, rstride=1, cstride=1, cmap='winter', edgecolor='none')
    ax.plot_surface(EDAD, ESCOLARIDAD, rendimientognp_L_promedio, rstride=1, cstride=1, cmap='hot', edgecolor='none')
    ax.plot_surface(EDAD, ESCOLARIDAD, rendimientogp_C_promedio, rstride=1, cstride=1, cmap= 'viridis' , edgecolor='none')
    ax.plot_surface(EDAD, ESCOLARIDAD, rendimientognp_C_promedio, rstride=1, cstride=1, cmap= 'plasma', edgecolor='none')
    ax.set_xlabel('edad')
    ax.set_ylabel('escolaridad')
    ax.set_zlabel(name)
    ax.view_init(elev=0, azim=30)
    plt.savefig(name_3dplot, bbox_inches='tight')
    plt.show()
    return(min_value, max_value, EDAD, ESCOLARIDAD, rendimientogp_L_promedio, rendimientognp_L_promedio, rendimientogp_C_promedio, rendimientognp_C_promedio)

#######################################
####################################### curvas de nivel 
def niveles_plot(name_niveles_plot, min_value, max_value, EDAD, ESCOLARIDAD, rendimientogp_L_promedio, rendimientognp_L_promedio, rendimientogp_C_promedio, rendimientognp_C_promedio):
    plt.subplot(1,2,1)
    plt.contourf(EDAD, ESCOLARIDAD, rendimientogp_L_promedio, 10, cmap='winter')
    plt.title('Mean Linear Model GP', fontsize=5)
    plt.xlabel('Age', fontsize=4)
    plt.ylabel('Studies', fontsize=4)
    plt.clim(min_value, max_value)
    plt.colorbar().set_label(a,fontsize=4, rotation = -90)

    plt.subplot(1,2,2)
    plt.contourf(EDAD, ESCOLARIDAD, rendimientognp_L_promedio, 10, cmap='winter')
    plt.title('Mean Linear Model GNP', fontsize=5)
    plt.xlabel('Age', fontsize=4)
    plt.ylabel('Studies', fontsize=4)
    plt.clim(min_value, max_value)
    plt.colorbar().set_label(a,fontsize=4, rotation = -90)

    #plt.subplot(2,2,3)
    #plt.contourf(EDAD, ESCOLARIDAD, rendimientogp_C_promedio, 10, cmap='winter')
    #plt.title('Modelo cuadratico promedio GP ', fontsize=7)
    #plt.xlabel('Edad', fontsize=4)
    #plt.ylabel('Escolaridad', fontsize=4)
    #plt.clim(min_value, max_value)
    #plt.colorbar().set_label(a,fontsize=4, rotation = -90)

    #plt.subplot(2,2,4)
    #plt.contourf(EDAD, ESCOLARIDAD, rendimientognp_C_promedio, 10, cmap='winter')
    #plt.title('Modelo cuadratico promedio GNP', fontsize=7)
    #plt.xlabel('Edad', fontsize=4)
    #plt.ylabel('Escolaridad', fontsize=4)
    #plt.clim(min_value, max_value)
    #plt.colorbar().set_label(a,fontsize=4, rotation = -90)

    plt.tight_layout(pad=3.0)
    plt.savefig(name_niveles_plot, bbox_inches='tight')
###########################################################
###########################################################

pruebas = ['MMSE', 'FLUIDEZ_SEM', 'DEN', 'APRENDIZAJE_LP', 'DIFERIDO_LP', 'RECONOCIMIENTO_LP', 'PRAX_COP', 'PRAX_EVO']

nombres_3dplot = ['MMSE.pdf', 'FLUIDEZ_SEM.pdf', 'DEN.pdf', 'APRENDIZAJE_LP.pdf', 'DIFERIDO_LP.pdf', 'RECONOCIMIENTO_LP.pdf',
                  'PRAX_COP.pdf', 'PRAX_EVO.pdf']

nombres_niveles_plot = ['MMSE_niveles.pdf', 'FLUIDEZ_SEM_niveles.pdf', 'DEN_niveles.pdf', 'APRENDIZAJE_LP_niveles.pdf', 
                        'DIFERIDO_LP_niveles.pdf', 'RECONOCIMIENTO_LP_niveles.pdf', 'PRAX_COP_niveles.pdf', 'PRAX_EVO_niveles.pdf']

nombres_Lcsv = ['MMSE_resume_L.csv', 'FLUIDEZ_SEM_resume_L.csv', 'DEN_resume_L.csv', 'APRENDIZAJE_LP_resume_L.csv', 
                'DIFERIDO_LP_resume_L.csv', 'RECONOCIMIENTO_LP_resume_L.csv', 'PRAX_COP_resume_L.csv', 'PRAX_EVO_resume_L.csv']

nombres_Ccsv = ['MMSE_resume_C.csv', 'FLUIDEZ_SEM_resume_C.csv', 'DEN_resume_C.csv', 'APRENDIZAJE_LP_resume_C.csv', 
                'DIFERIDO_LP_resume_C.csv', 'RECONOCIMIENTO_LP_resume_C.csv', 'PRAX_COP_resume_C.csv', 'PRAX_EVO_resume_C.csv']

f_normalizacion = [30, 33, 15, 30, 10, 10, 11, 11]

errores_al_cuadrado = pd.DataFrame(columns=['gp_L', 'gp_C', 'gnp_L', 'gnp_C'])

for i in range(0, len(pruebas )):
    a = pruebas[i]
    b = 'EDAD'
    c ='ESCOLARIDAD'
    predatos_sl = pd.read_csv('/home/martin/Desktop/neurociencias/modelo 1/simulacion/M1/eafit_junio_2022.csv') # base de datos

    predatos_gp = predatos_sl[predatos_sl['GRUPO'] == 'GP']   #esta es la base de datos original de los pasientes con el gen
    predatos_gp= predatos_gp[[a, b, c]]
    #datos_gp = pd.DataFrame()
    #datos_gp[a] = (predatos_gp[a]- predatos_gp[a].mean())/predatos_gp[a].std()
    #datos_gp[b] = predatos_gp[b]/55
    #datos_gp[c] = (predatos_gp[c] - 9.9)/predatos_gp[c].std()

    predatos_gnp = predatos_sl[predatos_sl['GRUPO'] == 'GNP']  #esta es la base de datos original de los pasientes sin el gen
    predatos_gnp= predatos_gnp[[a, b, c]]
    #datos_gnp = pd.DataFrame()
    #datos_gnp[a] = (predatos_gnp[a]-predatos_gnp[a].mean())/predatos_gnp[a].std()
    #atos_gnp[b] = predatos_gnp[b]/55
    #datos_gnp[c] = (predatos_gnp[c] - 9.9)/predatos_gnp[c].std()

    name_L = nombres_Lcsv [i]
    name_C = nombres_Ccsv [i]
    name = pruebas[i]
    name_3dplot = nombres_3dplot[i]
    name_niveles_plot = nombres_niveles_plot[i]
    parametrosgp_L_media, parametrosgnp_L_media, parametrosgp_C_media, parametrosgnp_C_media = parametros_info_df(predatos_gp, predatos_gnp, name_L, name_C, a, f_normalizacion[i], b, c)#
    sum_err = sum_dif_cuad(predatos_gp, predatos_gnp, name, parametrosgp_L_media, parametrosgnp_L_media, parametrosgp_C_media, parametrosgnp_C_media)
    errores_al_cuadrado = pd.concat([errores_al_cuadrado, sum_err], axis=0)
    min_value, max_value, EDAD, ESCOLARIDAD, rendimientogp_L_promedio, rendimientognp_L_promedio, rendimientogp_C_promedio, rendimientognp_C_promedio = plt_3d(name, name_3dplot, parametrosgp_L_media, parametrosgnp_L_media, parametrosgp_C_media, parametrosgnp_C_media )
    niveles_plot(name_niveles_plot, min_value, max_value, EDAD, ESCOLARIDAD, rendimientogp_L_promedio, rendimientognp_L_promedio, rendimientogp_C_promedio, rendimientognp_C_promedio)

errores_al_cuadrado.to_csv('errores_al_cuadrado.csv')