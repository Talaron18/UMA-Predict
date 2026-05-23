import pandas as pd
import numpy as np

class Expression:
    def eval(self, df): raise NotImplementedError
    def __add__(self, other): return Add(self, other)
    def __sub__(self, other): return Sub(self, other)
    def __mul__(self, other): return Mul(self, other)
    def __truediv__(self, other): return Div(self, other)
    def __radd__(self, other): return Add(other, self)
    def __rsub__(self, other): return Sub(other, self)
    def __rmul__(self, other): return Mul(other, self)
    def __rtruediv__(self, other): return Div(other, self)
    def __lt__(self, other): return BinaryOp(self, other, "lt")
    def __le__(self, other): return BinaryOp(self, other, "le")
    def __gt__(self, other): return BinaryOp(self, other, "gt")
    def __ge__(self, other): return BinaryOp(self, other, "ge")
    def __eq__(self, other): return BinaryOp(self, other, "eq")
    def __and__(self, other): return BinaryOp(self, other, "and")
    def __or__(self, other): return BinaryOp(self, other, "or")

class Feature(Expression):
    def __init__(self, name): self.name = name
    def eval(self, df): return df[self.name]
    def __repr__(self): return f"Feature('{self.name}')"

class Constant(Expression):
    def __init__(self, value): self.value = value
    def eval(self, df): return self.value
    def __repr__(self): return str(self.value)

def ensure_expr(x):
    if isinstance(x, Expression): return x
    if isinstance(x, str): return Feature(x)
    return Constant(x)

class BinaryOp(Expression):
    def __init__(self, left, right, op):
        self.left = ensure_expr(left)
        self.right = ensure_expr(right)
        self.op = op
    def eval(self, df):
        l = self.left.eval(df)
        r = self.right.eval(df)
        if self.op == "lt": return l < r
        elif self.op == "le": return l <= r
        elif self.op == "gt": return l > r
        elif self.op == "ge": return l >= r
        elif self.op == "eq": return l == r
        elif self.op == "and": return l & r
        elif self.op == "or": return l | r

class Add(Expression):
    def __init__(self, left, right): self.left, self.right = ensure_expr(left), ensure_expr(right)
    def eval(self, df): return self.left.eval(df) + self.right.eval(df)

class Sub(Expression):
    def __init__(self, left, right): self.left, self.right = ensure_expr(left), ensure_expr(right)
    def eval(self, df): return self.left.eval(df) - self.right.eval(df)

class Mul(Expression):
    def __init__(self, left, right): self.left, self.right = ensure_expr(left), ensure_expr(right)
    def eval(self, df): return self.left.eval(df) * self.right.eval(df)

class Div(Expression):
    def __init__(self, left, right): self.left, self.right = ensure_expr(left), ensure_expr(right)
    def eval(self, df): return self.left.eval(df) / (self.right.eval(df) + 1e-6)

class Where(Expression):
    def __init__(self, cond, true_val, false_val):
        self.cond, self.true_val, self.false_val = ensure_expr(cond), ensure_expr(true_val), ensure_expr(false_val)
    def eval(self, df):
        return pd.Series(np.where(self.cond.eval(df), self.true_val.eval(df), self.false_val.eval(df)), index=df.index)

class Bucket(Expression):
    def __init__(self, feature, bins, labels=None):
        self.feature, self.bins, self.labels = ensure_expr(feature), bins, labels
    def eval(self, df):
        return pd.cut(self.feature.eval(df), bins=self.bins, labels=self.labels, right=False).astype(float)

class HorseMean(Expression):
    def __init__(self, feature, window): self.feature, self.window = ensure_expr(feature), window
    def eval(self, df):
        return self.feature.eval(df).groupby(df["horse_id"]).shift(1).groupby(df["horse_id"]).rolling(self.window, min_periods=1).mean().reset_index(level=0, drop=True)

class HorseExpandingMean(Expression):
    def __init__(self, feature): self.feature = ensure_expr(feature)
    def eval(self, df):
        return self.feature.eval(df).groupby(df["horse_id"]).shift(1).groupby(df["horse_id"]).expanding().mean().reset_index(level=0, drop=True)

class HorseStd(Expression):
    def __init__(self, feature, window): self.feature, self.window = ensure_expr(feature), window
    def eval(self, df):
        return self.feature.eval(df).groupby(df["horse_id"]).shift(1).groupby(df["horse_id"]).rolling(self.window, min_periods=2).std().reset_index(level=0, drop=True)

class HorseExpandingStd(Expression):
    def __init__(self, feature): self.feature = ensure_expr(feature)
    def eval(self, df):
        return self.feature.eval(df).groupby(df["horse_id"]).shift(1).groupby(df["horse_id"]).expanding().std().reset_index(level=0, drop=True)

class FilterRollingMean(Expression):
    def __init__(self, feature, filter_expr, window):
        self.feature, self.filter_expr, self.window = ensure_expr(feature), ensure_expr(filter_expr), window
    def eval(self, df):
        f_val = self.filter_expr.eval(df)
        return self.feature.eval(df).groupby([df["horse_id"], f_val]).shift(1).groupby([df["horse_id"], f_val]).rolling(self.window, min_periods=1).mean().reset_index(level=[0, 1], drop=True)

class FilterExpandingMean(Expression):
    def __init__(self, feature, filter_expr): self.feature, self.filter_expr = ensure_expr(feature), ensure_expr(filter_expr)
    def eval(self, df):
        f_val = self.filter_expr.eval(df)
        return self.feature.eval(df).groupby([df["horse_id"], f_val]).shift(1).groupby([df["horse_id"], f_val]).expanding().mean().reset_index(level=[0, 1], drop=True)

class CSMean(Expression):
    def __init__(self, feature): self.feature = ensure_expr(feature)
    def eval(self, df): return self.feature.eval(df).groupby(df["race_id"]).transform("mean")

class CSZScore(Expression):
    def __init__(self, feature): self.feature = ensure_expr(feature)
    def eval(self, df):
        s = self.feature.eval(df)
        return s.groupby(df["race_id"]).transform(lambda x: (x - x.mean()) / (x.std() + 1e-6) if len(x) > 1 else 0.0)

class CSGroupCount(Expression):
    def __init__(self, category_expr): self.category_expr = ensure_expr(category_expr)
    def eval(self, df):
        cat = self.category_expr.eval(df)
        return df.groupby([df["race_id"], cat])["race_id"].transform("count")

class HorseHistoricalTopKMean(Expression):
    def __init__(self, target_feature, rank_feature, k=3):
        self.target, self.rank, self.k = ensure_expr(target_feature), ensure_expr(rank_feature), k
    def eval(self, df):
        target_s = self.target.eval(df)
        rank_s = self.rank.eval(df)
        temp_df = pd.DataFrame({'t': target_s, 'r': rank_s, 'hid': df['horse_id']})
        temp_df['t_shifted'] = temp_df.groupby('hid')['t'].shift(1)
        temp_df['r_shifted'] = temp_df.groupby('hid')['r'].shift(1)
        
        def top_k_mean(group):
            res, valid_t, valid_r = [], [], []
            for t, r in zip(group['t_shifted'], group['r_shifted']):
                if pd.notna(t) and pd.notna(r):
                    valid_t.append(t); valid_r.append(r)
                if len(valid_t) > 0:
                    idx = np.argsort(valid_r)[:self.k]
                    res.append(np.mean([valid_t[i] for i in idx]))
                else:
                    res.append(np.nan)
            return pd.Series(res, index=group.index)
        return temp_df.groupby('hid', group_keys=False).apply(top_k_mean)

horse_ave_3f_expr = (Feature("time_seconds") / Feature("distance")) * 600
pci_expr = (horse_ave_3f_expr - Feature("last_3f")) * 10 + 50
pace_ratio_expr = Feature("pace_first_half") / (Feature("last_3f") + 1e-6)
first_half_dev_expr = Feature("pace_first_half") - horse_ave_3f_expr

strategy_expr = Where(
    (first_half_dev_expr <= -1.0) & (pace_ratio_expr < 0.96), 0,
    Where(
        (first_half_dev_expr <= 0.2) & (pace_ratio_expr < 1.00), 1,
        Where((first_half_dev_expr > 0.2) & (first_half_dev_expr <= 1.2), 2, 3)
    )
)

bracket_cat_expr = Where(Feature("bracket_num") <= 4, 0, 1)
dist_cat_expr = Bucket("distance", bins=[0, 1400, 1900, 2400, 99999], labels=[0, 1, 2, 3])

HORSE_ALPHA = {
    "race_ave_3f":             CSMean(horse_ave_3f_expr),
    "race_last":               CSMean("last_3f"),
    "race_pci":                CSMean(pci_expr),
    "race_strategy_tendency":  CSMean(strategy_expr),
    "field_wear":              (Feature("race_no") - 1) * (Feature("weather_code") + Feature("condition_code")) / 6,
    "density":                 CSGroupCount(bracket_cat_expr),
    "horse_ave_3f_3":          HorseMean(horse_ave_3f_expr, window=3),
    "horse_ave_3f_all":        HorseExpandingMean(horse_ave_3f_expr),
    "horse_pci_3":             HorseMean(pci_expr, window=3),
    "horse_pci_all":           HorseExpandingMean(pci_expr),
    "passing_factor":          Feature("pace_second_half") - Feature("pace_first_half"),
    "weight_mean":             HorseExpandingMean("horse_weight"),
    "weight_alpha_3":          Feature("horse_weight") - HorseMean("horse_weight", window=3),
    "weight_best":             HorseHistoricalTopKMean(target_feature="horse_weight", rank_feature="finish_rank", k=3),
    "popularity_mean":         HorseExpandingMean("popularity"),
    "popularity_diff":         Feature("popularity") - HorseMean("popularity", window=3),
    "popularity_std_3":        HorseStd("popularity", window=3),
    "popularity_std_all":      HorseExpandingStd("popularity"),
    "bracket_suitability":     FilterExpandingMean("finish_rank", bracket_cat_expr),
    "distance_suitability":    FilterExpandingMean(horse_ave_3f_expr, dist_cat_expr),
    "field_suitability":       FilterExpandingMean("finish_rank", "distance")
}

class HorseAlphaEngine:
    def __init__(self, alpha_dict): self.alpha_dict = alpha_dict
    def compute(self, df):
        df = df.sort_values(["horse_id", "date"]).reset_index(drop=True)
        for alpha_name, expr in self.alpha_dict.items():
            df[alpha_name] = expr.eval(df)
        return df

if __name__ == "__main__":
    mock_data = {
        "race_id":           [201, 201, 202, 202, 203],
        "horse_id":          [1,   2,   1,   2,   1],
        "date":              pd.to_datetime(["2026-05-01", "2026-05-01", "2026-05-10", "2026-05-10", "2026-05-20"]),
        "race_no":           [1,   1,   3,   3,   5],
        "surface_code":      [0,   0,   1,   1,   0],
        "direction_code":    [1,   1,   1,   1,   1],
        "distance":          [1200, 1200, 1600, 1600, 2000],
        "weather_code":      [1,   1,   2,   2,   0],
        "condition_code":    [0,   0,   3,   3,   1],
        "gender_code":       [0,   1,   0,   1,   0],
        "age":               [3,   4,   3,   4,   3],
        "weight_carried":    [54,  56,  54,  57,  55],
        "time_seconds":      [71.2, 70.8, 96.5, 95.8, 122.4],
        "last_3f":           [34.8, 35.1, 36.2, 35.5, 34.1],
        "odds":              [2.8,  3.5,  4.2,  2.1,  3.0],
        "popularity":        [1,    2,    3,    1,    2],
        "horse_weight":      [480,  510,  484,  508,  482],
        "horse_weight_diff": [2,    -4,   4,    -2,   -2],
        "finish_rank":       [1,    2,    3,    1,    2],
        "bracket_num":       [2,    7,    3,    6,    4],
        "horse_num":         [3,    11,   4,    8,    5],
        "num_horses":        [12,   12,   14,   14,   10],
        "margin":            [0,    1.5,  2.0,  0,    0.5],
        "pace_first_half":   [36.4, 35.7, 60.3, 60.3, 88.3],
        "pace_second_half":  [34.8, 35.1, 36.2, 35.5, 34.1],
        "pace_diff":         [-1.6, -0.6, -24.1, -24.8, -54.2],
        "prize":             [1000, 1000, 1500, 1500, 2000],
        "career_races":      [12,   20,   13,   21,   14],
        "career_wins":       [3,    5,    3,    6,    3]
    }
    
    df = pd.DataFrame(mock_data)
    engine = HorseAlphaEngine(HORSE_ALPHA)
    result = engine.compute(df)
    
    cols = ["race_id", "horse_id", "field_wear", "density", "passing_factor", "weight_best", "distance_suitability"]
    print(result[cols].to_string())