"""Dataframe conversion between Pandas and PySpark.

The original conversion code is here:

```python
from pyspark.sql.pandas.conversion import PandasConversionMixin, SparkConversionMixin
```
"""

import pandas as pd
from pyspark.sql import DataFrame as SparkDF, SparkSession
from pyspark.sql.types import StructType, StructField

from pyspark_mc.dist import Distribution
from pyspark_mc.dist.base import DISTRO_KEY, to_distribution


_old_toPandas = SparkDF.toPandas

_old_createDataFrame = SparkSession.createDataFrame


def toPandas(sdf: SparkDF) -> pd.DataFrame:
    """Converts the Spark dataframe to Pandas, including distribution types."""
    # Get fields
    distro_map: dict[str, str] = {}
    for fld in sdf.schema.fields:
        dist_name = fld.metadata.get(DISTRO_KEY)
        if dist_name is not None:
            distro_map[fld.name] = dist_name
    # Convert within Spark
    pdf: pd.DataFrame = _old_toPandas(sdf)
    # Add extra conversions
    for col, dist_name in distro_map.items():
        pdf[col] = pdf[col].apply(lambda x: to_distribution(dist_name, *x))
    return pdf


def createDataFrame(
    spark: SparkSession,
    data: pd.DataFrame,
    schema=None,
    samplingRatio=None,
    verifySchema=True,
) -> SparkDF:
    """Converts the Pandas dataframe to Spark."""
    if isinstance(data, pd.DataFrame):
        pdf = data
        # Infer from the first row
        tmp = _old_createDataFrame(spark, pdf.iloc[0:1], schema=schema)
        inferred_fields = tmp.schema.fields
        cols = [x.name for x in inferred_fields]
        pdf = pdf[cols].copy()

        # Get Distribution fields
        new_fields = []
        for infer_fld, col in zip(inferred_fields, cols):
            v = pdf[col].iloc[0]
            if isinstance(v, Distribution):
                fld = v.as_struct_field(col)
            else:
                fld = infer_fld
            new_fields.append(fld)

        schema = StructType(new_fields)
    res = _old_createDataFrame(
        spark,
        data,
        schema=schema,
        samplingRatio=samplingRatio,
        verifySchema=verifySchema,
    )
    return res


def install_converters():
    """Monkey-patches to install converters to SparkDF and SparkSession."""
    toPandas.__doc__ = SparkDF.toPandas.__doc__
    SparkDF.toPandas = toPandas

    createDataFrame.__doc__ == SparkSession.createDataFrame.__doc__
    SparkSession.createDataFrame = createDataFrame
