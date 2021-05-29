import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io.arff import loadarff
from utils import bytesToString
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler, Normalizer, LabelEncoder, OneHotEncoder

# NumPy Data types
CATEGORICAL = 'O'

# String Conversion Methods
INT = 'int'
ONEHOT = 'onehot'
STRING_CONVERSION = [INT, ONEHOT]

# Float Normalization Methods
MIN_MAX = "min-max"
STANDARISATION = "stardardisation"
MEAN = "mean"
UNIT = "unit"
FLOAT_NORMALIZATION = [MIN_MAX, STANDARISATION, MEAN, UNIT]

# Missing Data Imputation Methods
IMPUTE_MOST_FREQUENT = "most_frequent"
IMPUTE_MEDIAN = "median"
IMPUTE_MEAN = "mean"
IMPUTE_CONSTANT = "constant"
RANDOM = "random"
LINEAR_REGRESSION = "linear_regression"
LOGISTIC_REGRESSION = "logistic_regression"
MISSING_DATA_IMPUTATION = [IMPUTE_MOST_FREQUENT, IMPUTE_MEDIAN, IMPUTE_MEAN,
                           IMPUTE_CONSTANT, RANDOM, LINEAR_REGRESSION, LOGISTIC_REGRESSION]

class PandasDataset:
    """
    Class responsible of loading and formatting the csv file data

    input:
        args:
            csv: Path object with information on the path to csv

    """
    def __init__(self, csvPath, stringConversion=INT, floatNormalization=MIN_MAX, missingDataImputation=IMPUTE_MOST_FREQUENT):
        self.path = Path(csvPath)
        self.assertInitParameters(stringConversion, floatNormalization, missingDataImputation)

        self.stringConversion = stringConversion
        self.floatNormalization = floatNormalization
        self.missingDataImputation = missingDataImputation
        self.labelEncoders = {}
        self.data_ = pd.read_csv(csvPath)
        self.formatDataFrame()
        self.targetLabel = self.data_.columns.to_list()[-1]

    def assertInitParameters(self, stringConversion, floatNormalization, missingDataImputation):
        assert self.path.suffix == '.csv', f'dataset must be a csv file, not {self.path.suffix}'
        assert stringConversion in STRING_CONVERSION, f"{stringConversion} is not a valid value for stringConversion"
        assert floatNormalization in FLOAT_NORMALIZATION, f"{floatNormalization} normalization type not supported"
        assert missingDataImputation in MISSING_DATA_IMPUTATION, f"{missingDataImputation} is not supported"

    def formatDataFrame(self):
        self.data_ = pd.DataFrame(self.data_)
        # self.data_ = data.applymap(bytesToString) # apply type conversion to all items in DataFrame
        self.formatColumns()

    def formatColumns(self):
        columnTypes = {}
        columnNames = self.data_.columns.to_list()
        for column in self.data_.columns[:-1].to_list():
            columnData = self.data_[column].copy()
            columnTypes[column] = columnData.dtype.kind
            self.preprocessDataTypes(column, columnData, columnNames)
        for column in self.data_.columns[:-1].to_list():
            self.preprocessDataMissingValues(column, columnTypes[column], columnNames)
            self.preprocessDataRanges(column, columnTypes[column])
        self.formatGoldStandard(columnNames[-1])

    def preprocessDataTypes(self, column, columnData, columnNames):
        if columnData.dtype.kind == CATEGORICAL:
            self.formatCategoricalData(column, columnData, columnNames)

    def preprocessDataRanges(self, column, columnType):
        if not (columnType == CATEGORICAL and self.stringConversion == ONEHOT):
            self.data_[column] = self.normalizeFloatColumn(self.data_[column])

    def preprocessDataMissingValues(self, column, columnType, columnNames):
        if columnType == CATEGORICAL:
            if self.stringConversion == INT:
                if '?' in self.labelEncoders[column].classes_:
                    missingValue = self.labelEncoders[column].transform(["?"])[0]
                    self.data_[column] = self.fixMissingValues(column, self.data_[column], columnNames, missingValue)

    def formatGoldStandard(self, gsColumn):
        self.data_[gsColumn] = self.convertStringsToInt(gsColumn, self.data_[gsColumn])

    def formatCategoricalData(self, column, columnData, columnNames):
        if self.stringConversion == INT:
            self.data_[column] = self.convertStringsToInt(column, columnData)
        elif self.stringConversion == ONEHOT:
            self.data_ = self.convertStringToOH(column, columnData, columnNames)

    def fixMissingValues(self, column, columnData, columnNames, missingValue):
        if self.missingDataImputation == RANDOM:
            return self.randomMissingValues(columnData, missingValue)
        elif self.missingDataImputation in [IMPUTE_MOST_FREQUENT, IMPUTE_MEDIAN, IMPUTE_MEAN, IMPUTE_CONSTANT]:
            return self.imputerMissingValues(columnData, missingValue)
        elif self.missingDataImputation in [LINEAR_REGRESSION, LOGISTIC_REGRESSION]:
            return self.regressionMissingValues(column, columnData, columnNames, missingValue)

    def randomMissingValues(self, columnData, missingValue):
        columnData[columnData == missingValue] = np.random.choice(columnData[columnData != missingValue],
                                                         columnData[columnData == missingValue].count())
        return columnData

    def imputerMissingValues(self, columnData, missingValue):
        columnData = SimpleImputer(missing_values=missingValue, strategy=self.missingDataImputation) \
                    .fit_transform(columnData.values.reshape(-1, 1))
        return columnData.squeeze()

    def regressionMissingValues(self, column, columnData, columnNames, missingValue):
        model = LogisticRegression(n_jobs=-1) if self.missingDataImputation == LOGISTIC_REGRESSION \
                                              else LinearRegression(n_jobs=-1)
        features = list(set(columnNames[:-1]) - set(column))
        allOtherColumnsData = self.data_[features]
        model.fit(X=allOtherColumnsData[columnData != missingValue], y=columnData[columnData != missingValue])
        columnData[columnData == missingValue] = model.predict(allOtherColumnsData[columnData == missingValue])
        return columnData

    def convertStringsToInt(self, column, columnData):
        self.labelEncoders[column] = LabelEncoder()
        return self.labelEncoders[column].fit_transform(columnData).astype(np.int)

    def convertStringToOH(self, column, columnData, columnNames):
        self.labelEncoders[column] = OneHotEncoder(sparse=False, handle_unknown='ignore')
        oneHotVectors = self.labelEncoders[column].fit_transform(columnData.reshape(-1, 1)).astype(np.int)
        OHColumnNames = [f"{column}_{columnValue}" for columnValue in self.labelEncoders[column].categories_[0]]
        OHDataFrame = pd.DataFrame(oneHotVectors, columns=OHColumnNames)
        columnNames[columnNames.index(column):columnNames.index(column)+1] = OHDataFrame.columns.to_list()
        return self.data_.drop(column, axis=1).join(OHDataFrame, how='outer')[columnNames]

    def normalizeFloatColumn(self, data):
        if self.floatNormalization == STANDARISATION:
            return scale(data)
        elif self.floatNormalization == MEAN:
            scaler = StandardScaler()
        elif self.floatNormalization == MIN_MAX:
            scaler = MinMaxScaler()
        elif self.floatNormalization == UNIT:
            scaler = Normalizer()
        data = np.array(data).reshape(-1, 1)
        data = scaler.fit_transform(data)
        return pd.Series(data.reshape(-1))

    @property
    def data(self):
        return self.data_

    @property
    def x(self):
        return self.data_.drop(self.targetLabel, axis=1).to_numpy()

    @property
    def y(self):
        return self.data_[self.targetLabel].to_numpy()

    @property
    def name(self):
        return self.path.stem

    def getStringMapData(self):
        return self.labelEncoders


class PandasUnsupervised(PandasDataset):
    def formatColumns(self):
        columnTypes = {}
        columnNames = self.data_.columns.to_list()
        for column in columnNames:
            columnData = self.data_[column].copy()
            columnTypes[column] = columnData.dtype.kind
            self.preprocessDataTypes(column, columnData, columnNames)
        for column in columnNames:
            self.preprocessDataMissingValues(column, columnTypes[column], columnNames)
            self.preprocessDataRanges(column, columnTypes[column])
