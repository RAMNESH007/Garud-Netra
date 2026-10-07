# Garud-Netra

## AI-Powered Bitcoin Transaction Monitoring & Investigation

Garud-Netra is an offline AI-assisted investigation system for analyzing
Bitcoin transaction and related network observations.

The system combines:

- Data ingestion
- Transaction preprocessing
- Network preprocessing
- Transaction/network correlation
- Transaction feature engineering
- Network feature engineering
- Combined feature engineering
- Transaction graph analysis
- Anomaly detection
- Entity clustering
- Investigative risk scoring
- Explainability
- Offline investigation dashboard
- Post-Quantum Cryptography (PQC) evidence protection
- Automated security and PQC testing

> Garud-Netra is an investigative analysis system. Model outputs represent
> evidence signals and investigation priorities; they do not establish
> criminality or wrongdoing.

---

## Architecture

```text
DATASET
   |
   v
DATA INGESTION
   |
   v
PREPROCESSING
   |
   +-----------------------------+
   |                             |
   v                             v
TRANSACTION DATA             NETWORK DATA
   |                             |
   +-------------+---------------+
                 |
                 v
       TRANSACTION + NETWORK
           CORRELATION
                 |
                 v
        FEATURE ENGINEERING
                 |
                 v
        TRANSACTION GRAPH
                 |
          +------+------+
          |      |      |
          v      v      v
        ANOMALY CLUSTER RISK
        MODEL   MODEL   SCORING
          |      |      |
          +------+------+
                 |
                 v
          EXPLAINABILITY
                 |
                 v
        INVESTIGATION RESULT
                 |
        +--------+--------+
        |                 |
        v                 v
    DASHBOARD          PQC SECURITY
                           |
                    +------+------+
                    |             |
                    v             v
                  SIGN          VERIFY
                    |             |
                    +------+------+
                           |
                           v
                  OFFLINE CASE
                    PACKAGE
