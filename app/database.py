import sqlite3
from datetime import datetime

DB_NAME = "predictions.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        AccountWeeks INTEGER,
        ContractRenewal INTEGER,
        DataPlan INTEGER,
        DataUsage REAL,
        CustServCalls INTEGER,
        DayMins REAL,
        DayCalls INTEGER,
        MonthlyCharge REAL,
        OverageFee REAL,
        RoamMins REAL,
        churn_probability REAL,
        prediction INTEGER,
        risk_level TEXT
    )
    """)

    conn.commit()
    conn.close()


def log_prediction(data, probability, prediction, risk_level):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO predictions (
        timestamp,
        AccountWeeks,
        ContractRenewal,
        DataPlan,
        DataUsage,
        CustServCalls,
        DayMins,
        DayCalls,
        MonthlyCharge,
        OverageFee,
        RoamMins,
        churn_probability,
        prediction,
        risk_level
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        data["AccountWeeks"],
        data["ContractRenewal"],
        data["DataPlan"],
        data["DataUsage"],
        data["CustServCalls"],
        data["DayMins"],
        data["DayCalls"],
        data["MonthlyCharge"],
        data["OverageFee"],
        data["RoamMins"],
        probability,
        prediction,
        risk_level
    ))

    conn.commit()
    conn.close()
    
def get_stats():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 1")
    high_risk = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(churn_probability) FROM predictions")
    avg_prob = cursor.fetchone()[0]

    conn.close()

    if total == 0:
        return {
            "total_predictions": 0,
            "high_risk_predictions": 0,
            "high_risk_percentage": 0,
            "average_churn_probability": 0
        }

    return {
        "total_predictions": total,
        "high_risk_predictions": high_risk,
        "high_risk_percentage": round((high_risk / total) * 100, 2),
        "average_churn_probability": round(avg_prob, 4)
    }