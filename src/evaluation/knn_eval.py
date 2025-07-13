import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def evalute_knn(X_train, X_test, y_train, y_test, n_neighbors=5):
    """
        Avaliação do desempenho de classificador KNN, com retorno de métricas de avaliação e tempo de execução.

        Args:
            X_train (array-like): Dados de treinamento.
            X_test (array-like): Dados de teste.
            y_train (array-like): Rótulos de treinamento.
            y_test (array-like): Rótulos de teste.
            n_neighbors (int): Número de vizinhos a serem considerados pelo KNN.

        Returns:
            (f1_score_macro, accuracy, precision_macro, recall_macro, tempo_execução):

    """
    start_time = time.time()

    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    end_time = time.time()

    f1 = f1_score(y_test, y_pred, average="macro")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    execution_time = end_time - start_time
    
    return f1, accuracy, precision, recall, execution_time