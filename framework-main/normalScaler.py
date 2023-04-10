class NormalScaler:
    def fit(self, X):
        self.min = X.min(axis=0)
        self.max = X.max(axis=0) - self.min

    def transform(self, X):
        return (X - self.min) / self.max

    def inverse_transform(self, X_scaled):
        return X_scaled * self.max + self.min