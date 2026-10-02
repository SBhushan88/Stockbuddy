"""
tools.py  -  the "calculator" part of the chatbot.

Why this file exists:
AI models are good at language but can make mistakes with numbers.
So all stock lookups and maths are done HERE, by normal Python code.
Gemini only decides WHICH function to call and then explains the result.
"""

import math
import difflib
import pandas as pd

CSV_FILE = "inventory.csv"
SAFETY_DAYS = 7    # extra days of stock we like to keep as a cushion
COVER_DAYS = 30    # when we reorder, we order enough for this many days


def load_data() -> pd.DataFrame:
    """Read the product sheet (inventory.csv) and add the sales-speed columns."""
    df = pd.read_csv(CSV_FILE)
    # average units sold per month over the last 3 months
    df["avg_monthly_sales"] = df[["sales_jul", "sales_aug", "sales_sep"]].mean(axis=1)
    # units sold per day (assume a 30-day month)
    df["daily_sales"] = df["avg_monthly_sales"] / 30
    return df


def _find(name: str, df: pd.DataFrame):
    """Find products whose name matches what the user typed (spelling can be rough)."""
    q = name.strip().lower()
    names = df["product"].str.lower()
    # 1) the typed text appears inside a product name
    hits = df[names.str.contains(q, regex=False)]
    # 2) every word typed appears in the product name
    if hits.empty:
        words = q.split()
        mask = names.apply(lambda n: all(w in n for w in words))
        hits = df[mask]
    # 3) spelling is slightly off -> closest match
    if hits.empty:
        close = difflib.get_close_matches(q, names.tolist(), n=3, cutoff=0.6)
        hits = df[names.isin(close)]
    return hits


def _status(row) -> str:
    if row["stock_qty"] == 0:
        return "OUT OF STOCK"
    if row["stock_qty"] <= row["reorder_level"]:
        return "LOW STOCK"
    return "OK"


def _lookup(product_name: str):
    """Return (row, None) if exactly one match, otherwise (None, message_dict)."""
    df = load_data()
    hits = _find(product_name, df)
    if hits.empty:
        return None, {"result": "not_found",
                      "message": f"No product matching '{product_name}' in the inventory.",
                      "available_products": df["product"].tolist()}
    if len(hits) > 1:
        return None, {"result": "ambiguous",
                      "message": "More than one product matches. Ask the user which one they mean.",
                      "matches": hits["product"].tolist()}
    return hits.iloc[0], None


# ----------------------------------------------------------------------
# The functions below are the "tools" that Gemini is allowed to call.
# The text inside the triple quotes tells Gemini what each tool does.
# ----------------------------------------------------------------------

def get_stock(product_name: str) -> dict:
    """Get the current stock, price, reorder level and status of ONE product.
    Use this when the user asks how much of a product is available."""
    row, problem = _lookup(product_name)
    if problem:
        return problem
    return {
        "product": row["product"], "category": row["category"],
        "stock_qty": int(row["stock_qty"]), "reorder_level": int(row["reorder_level"]),
        "unit_price_inr": int(row["unit_price"]), "supplier": str(row["supplier"]),
        "status": _status(row),
    }


def list_low_stock() -> dict:
    """List every product that is low on stock or out of stock (stock at or below its reorder level).
    Use this for questions like 'what is running low?' or 'any stock-outs?'."""
    df = load_data()
    low = df[df["stock_qty"] <= df["reorder_level"]]
    items = [{"product": r["product"], "stock_qty": int(r["stock_qty"]),
              "reorder_level": int(r["reorder_level"]), "status": _status(r)}
             for _, r in low.iterrows()]
    return {"count": len(items), "items": items}


def reorder_suggestion(product_name: str) -> dict:
    """Suggest whether to reorder ONE product and how many units, based on how fast it sells.
    Use this when the user asks 'should I reorder X?' or 'how much X should I order?'."""
    row, problem = _lookup(product_name)
    if problem:
        return problem
    return _reorder_calc(row)


def _reorder_calc(row) -> dict:
    daily = row["daily_sales"]
    stock = int(row["stock_qty"])
    lead = int(row["lead_time_days"])
    base = {"product": row["product"], "stock_qty": stock,
            "avg_monthly_sales": round(float(row["avg_monthly_sales"]), 1),
            "lead_time_days": lead, "supplier": row["supplier"]}
    if daily == 0:
        return {**base, "needs_reorder": False, "suggested_order_qty": 0,
                "reason": "No sales in the last 3 months (possible dead stock). Do not reorder; consider a discount."}
    days_cover = stock / daily
    reorder_point = daily * (lead + SAFETY_DAYS)
    needs = stock <= reorder_point or stock <= row["reorder_level"]
    qty = max(0, math.ceil(daily * (lead + COVER_DAYS) - stock)) if needs else 0
    return {**base, "daily_sales": round(float(daily), 1), "days_of_stock_left": round(float(days_cover), 1),
            "needs_reorder": bool(needs), "suggested_order_qty": qty,
            "formula": f"order = daily sales x (lead time {lead} + {COVER_DAYS} cover days) - current stock"}


def list_reorder_suggestions() -> dict:
    """List ALL products that need reordering now, with suggested quantities, most urgent first.
    Use this for 'what should I order this week?' or 'give me a purchase list'."""
    df = load_data()
    out = []
    for _, r in df.iterrows():
        calc = _reorder_calc(r)
        if calc["needs_reorder"]:
            out.append(calc)
    out.sort(key=lambda c: c["days_of_stock_left"])
    return {"count": len(out), "items": out}


def top_sellers(n: int = 5) -> dict:
    """Show the top N best-selling products by average monthly units sold (default 5)."""
    df = load_data().sort_values("avg_monthly_sales", ascending=False).head(max(1, min(int(n), 24)))
    return {"items": [{"product": r["product"], "avg_monthly_sales": round(float(r["avg_monthly_sales"]), 1)}
                      for _, r in df.iterrows()]}


def list_dead_stock() -> dict:
    """List products that have had ZERO sales in the last 3 months but still have stock (dead stock),
    with the money tied up in them. Use for 'which items have not sold?' or 'slow or dead stock'."""
    df = load_data()
    dead = df[(df["avg_monthly_sales"] == 0) & (df["stock_qty"] > 0)]
    items = [{"product": r["product"], "stock_qty": int(r["stock_qty"]),
              "value_tied_up_inr": int(r["stock_qty"] * r["unit_price"])} for _, r in dead.iterrows()]
    return {"count": len(items), "items": items}


def stock_value_summary() -> dict:
    """Total value of stock currently held (stock x unit price), overall and by category."""
    df = load_data()
    df["value"] = df["stock_qty"] * df["unit_price"]
    return {"total_value_inr": int(df["value"].sum()),
            "by_category_inr": {k: int(v) for k, v in df.groupby("category")["value"].sum().items()}}


# the list of tools we hand over to Gemini
TOOLS = [get_stock, list_low_stock, reorder_suggestion,
         list_reorder_suggestions, top_sellers, stock_value_summary, list_dead_stock]
