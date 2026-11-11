# Superstore demand brief

**Forecast window:** Jan 07, 2018 – Feb 25, 2018  
**History through:** Dec 31, 2017  
**Model used:** Random Forest (holdout MAE $7,061, MAPE 44.6%)

## What the forecast says

We expect about **$101,227** in sales over the next 8 weeks.

That is **-42.3%** versus the last 8 (holiday) weeks of actuals ($175,501), and **+78.4%** versus the same 8 weeks a year earlier ($56,730). The busiest predicted week is Jan 07 (~$13,477); the quietest is Feb 11 (~$11,927).

This pattern matches Superstore’s calendar: November–December weeks average about $19,516, while January–February weeks average about $5,073. After the holiday spike, demand cools. Plan for a softer first quarter than December, but a stronger start than the 2014–2016 winters because the business has been growing.

## How to use it this week

- **Inventory.** Do not restock holiday-peak quantities into January. Lean into Office Supplies (largest order count) for steady replenishment; keep furniture and technology closer to the forecast band so cash is not sitting in slow movers.
- **Cash.** The next 8 weeks total ~$101,227. If last year’s Q1 conversion of sales-to-cash holds, this is the receipts envelope for rent, suppliers, and payroll. Stress-test a buffer of about $7,061 per week (the model’s typical miss) before committing to big POs.
- **Staffing.** Peak weeks in this window still need floor and shipping coverage, but fewer extra seasonal hours than November–December. Schedule down toward the quieter weeks rather than keeping holiday headcount.

## How good is the number?

On the last 12 weeks (mostly holiday peak), the chosen model’s average miss was **$7,061 per week** (44.6% MAPE). That beat last-year-same-week ($8,375 MAE). Percentage error looks high because December weeks swing hard; use the dollar MAE as the planning buffer. If a promotion, store opening, or stockout is coming that did not exist in 2014–2017, override the model.

## Caveats

- History is 2014–2017 U.S. Superstore orders. There is no COVID shock in this file, and there are no 2018+ prices.
- One national weekly total hides regional mix (West and East dominate). A regional manager should split the same pipeline by `Region` before buying local inventory.
- Empty weeks were treated as true zeros, not missing data.
- The shaded range is ± the holdout MAE, not a statistical guarantee.

## Bottom line

Treat **$101,227 over 8 weeks** as the planning midpoint. Buy and staff below holiday-peak levels, keep a buffer up to the top of the band, and revisit the forecast after the first two actual weeks of the new year.
