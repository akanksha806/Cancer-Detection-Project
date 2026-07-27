# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# import pandas as pd
# from sklearn.impute import SimpleImputer



# class Data:
#     # read the dataset
#     def preprocess_Data(self,x):
#         # df = pd.read_csv(path)

#         # # seperating the dependent and independent columns
#         # x = df.drop(['id','Unnamed:32'],axis=1)
#         # y = df['diagnosis']

#         # seperate categorical and numerical column
#         cat_col = x.select_dtypes(include='object').columns
#         num_col = x.select_dtypes(exclude='object').columns

#         # creating pipeline for the categorigal and numerical columns
#         cat_pipeline = Pipeline(steps=[
#             ('imputer',SimpleImputer(strategy='most_frequent')),
#             ('encoder',OneHotEncoder(handle_unknown='ignore'))
#         ])
#         num_pipeline = Pipeline(steps=[
#             ('imputer',SimpleImputer(strategy='median')),
#             ('scaler',StandardScaler())
#             ])

#         # columns transformer for both pipelines
#         preprocessor = ColumnTransformer(transformers=[
#         ("num", num_pipeline, num_col),
#         ("cat", cat_pipeline, cat_col)
#     ])

#     preprocessed_data = preprocessor.fit_transform(x)

#     return preprocessed_data
#     #missing value
#     # scalling
#     # encoding
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


class Data:

    def preprocess_data(self, x):

        # Separate categorical and numerical columns
        cat_col = x.select_dtypes(include='object').columns
        num_col = x.select_dtypes(exclude='object').columns

        # Numerical Pipeline
        num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Categorical Pipeline
        cat_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, num_col),
        ("cat", cat_pipeline, cat_col)
        ])
        return preprocessor

  