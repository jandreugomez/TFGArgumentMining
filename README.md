# TFGArgumentMining

# Preprocesado de Datos

Para generar los datos al formato y con las etiquetas necesarias del corpus original en ingles, es necesario ejecutar esto:

```
sh Procesador_datos.sh
```

En el caso de generar el fichero de las relaciones es necesario ejecutar antes la traduccion del corpus ya que las relaciones se generan a partir del conjunto de datos traducido.

# Traduccion

Para ejecutar las traducciones del corpus es necesario ejecutar lo siguiente:

```
sh Traductor.sh
```

# Inferencia de anotaciones

Para ejecutar la inferencia de componentes al nuevo corpus es necesario ejecutar lo siguiente:

```
sh AnotadorComponentes.sh
```

Para ejecutar la inferencia de relaciones al nuevo corpus es necesario ejecutar lo siguiente:

```
sh AnotadorRelaciones.sh
```
