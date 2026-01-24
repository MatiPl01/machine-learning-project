## Aktualna lista modeli

| Model | Typ | Złożoność | Plik |
|-------|-----|-----------|------|
| GOAT | Transformer | O(N) | `models/goat.py` |
| Exphormer | Transformer | O(Nd) | `models/exphormer.py` |
| GCN | Baseline | O(E) | `models/baselines.py` |
| GAT | Baseline | O(E) | `models/baselines.py` |
| **GIN** | Baseline | O(E) | `models/baselines.py` |
| GraphMLP | Baseline | O(N) | `models/baselines.py` |
| **GCNVirtualNode** | Hybrid | O(E+N) | `models/hybrid.py` |
| **GINVirtualNode** | Hybrid | O(E+N) | `models/hybrid.py` |

## Aktualna lista datasetów

| Dataset | Zadanie | Rozmiar | Metryka |
|---------|---------|---------|---------|
| ZINC | Regression | 12K | MAE |
| ogbg-molhiv | Binary class. | 41K | ROC-AUC |
| **ogbg-molpcba** | Multi-label | 438K | AP |
| **ogbg-ppa** | Multi-class | 158K | Accuracy |
| peptides-func | Multi-label | 15K | AP |


Co trzeba jeszcze zrobić? Uruchomić jakieś porównanie na gpu (nie wiem czy notebook  experiments/compare_all_models.ipynb wystarczy, jak tak to super) i to wszystko opisać  w formie artykułu w readme? No czy właśnie w jakiej formie wsm? Można ten opis jakiś taki se zrobić na wtorek i dostać feedback.
