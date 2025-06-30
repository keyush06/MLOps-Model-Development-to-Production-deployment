from sklearn.feature_extraction import DictVectorizer

def create_feature_matrix(df, dv=None, fit_dv=True):
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    if fit_dv:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    
    return X, dv
