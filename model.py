from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


CAT_COLS = [
    "Employment_Info_3",
    "Employment_Info_5",
    "InsuredInfo_1",
    "InsuredInfo_2",
    "InsuredInfo_5",
    "InsuredInfo_6",
    "InsuredInfo_7",
    "Insurance_History_2",
    "Family_Hist_1",
    "Medical_History_4",
    "Medical_History_5",
    "Medical_History_6",
    "Medical_History_9",
    "Medical_History_12",
    "Medical_History_13",
    "Medical_History_16",
    "Medical_History_17",
    "Medical_History_18",
    "Medical_History_20",
    "Medical_History_21",
    "Medical_History_23",
    "Medical_History_27",
    "Medical_History_28",
    "Medical_History_29",
    "Medical_History_30",
    "Medical_History_33",
    "Medical_History_38",
    "Medical_History_39",
    "Medical_History_40",
]



NUM_COLS = [
    "Product_Info_4",
    "Ins_Age",
    "Ht",
    "Wt",
    "BMI",
    "Family_Hist_2",
    "Family_Hist_4",
    "Medical_History_1",
    "Medical_History_10",
    "Medical_History_15",
    "Medical_History_32",
    "Medical_Keyword_1",
    "Medical_Keyword_3",
    "Medical_Keyword_10",
    "Medical_Keyword_15",
    "Medical_Keyword_22",
    "Medical_Keyword_23",
    "Medical_Keyword_24",
    "Medical_Keyword_25",
    "Medical_Keyword_37",
    "Medical_Keyword_38",
    "Medical_Keyword_40",
    "Medical_Keyword_42",
    "Medical_Keyword_43",
    "Medical_Keyword_46",
    "Medical_Keyword_47",
    "Medical_Keyword_48",
]



def build_model():
    
    preprocessor = ColumnTransformer(
    [
        (
        "cat_cols",
        OneHotEncoder(sparse=False, handle_unknown="ignore"),
        CAT_COLS,
        ),
        ("cont_cols", "passthrough", NUM_COLS),
    ]
    )
    
    model = XGBClassifier(
    max_depth=7,
    min_child_weight=5,
    n_estimators=50,
    colsample=0.8,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])




