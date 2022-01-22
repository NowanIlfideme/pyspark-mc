"""Basic distributions."""

from pyspark.sql.types import (
    StructField,
    StructType,
    ArrayType,
    FloatType,
    DoubleType,
    StringType,
    IntegerType,
)

if True:
    FT = FloatType()
else:
    FT = DoubleType()

name = StructField(name="name", dataType=StringType(), nullable=False)

# Basic
Constant = StructType([name, StructField("value", FT, nullable=False)])

# Continuous distributions
Uniform = StructType(
    [
        name,
        StructField("lower", FT, nullable=False),
        StructField("upper", FT, nullable=False),
    ]
)
Normal = StructType(
    [
        name,
        StructField("mu", FT, nullable=False),
        StructField("sd", FT, nullable=True),
        StructField("tau", FT, nullable=True),
    ]
)

# Discrete distributions
Bernoulli = StructType(
    [
        name,
        StructField("p", FT, nullable=False),
    ]
)
Categorical = StructType(
    [
        name,
        StructField("values", ArrayType(FT, False), nullable=False),
        StructField("probs", ArrayType(FT, False), nullable=False),
    ]
)
DiscreteUniform = StructType(
    [name, StructField("values", ArrayType(FT, False), nullable=False)]
)
Poisson = StructType([name, StructField("mu", FT, nullable=False)])
Binomial = StructType(
    [
        name,
        StructField("p", FT, nullable=False),
        StructField("n", IntegerType(), nullable=False),
    ]
)
Empirical = StructType(
    [name, StructField("draws", ArrayType(FT, False), nullable=False)]
)

# Multivariate distributions
Dirichlet = StructType([[name, StructField("a", ArrayType(FT, False), nullable=False)]])
