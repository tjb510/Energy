# Energy
Repository for 2025 Spring project on Single Clearing Price Auctions

# Investigating California's Single-Clearing Price Auction System for Energy Markets

## Motivation
We are investigating whether the current market structure, the **Single Clearing Price (SCP) auction**, is subject to market manipulation via strategic bidding by energy suppliers. 

California has the **2nd highest residential energy prices** in the entire country and holds the title of the **highest residential energy prices in the lower 48**. With consumers facing increasing prices in all markets, it is in our best interest to investigate potential reasons for California's higher-than-average electricity prices. 

We also want to help **dispel the notion that renewables are the cause of these higher prices**, despite their ever-plummeting costs. To do this, we aim to investigate whether the **current structure of the California electricity market** lends itself to being gamed by generators **to the detriment of ratepayers downstream**.

## Energy Market Structure
To quote Mark Christie:

> “One of the most succinct and understandable descriptions of single-clearing price mechanisms and how they work in power markets is found in a U.S. Supreme Court opinion written by Justice Elena Kagan. It is worth quoting liberally herein.”

Justice Kagan describes SCP auctions as follows:

> “These wholesale auctions serve to balance supply and demand on a continuous basis, producing prices for electricity that reflect its value at given locations and times throughout each day. Such a real-time mechanism is needed because, unlike most products, electricity cannot be stored effectively. Suppliers must generate—every day, hour, and minute—the exact amount of power necessary to meet demand from the utilities and other **load-serving entities (LSEs)** that buy power at wholesale for resale to users. 
>
> To ensure that happens, wholesale market operators obtain:  
> **(1)** Orders from LSEs indicating how much electricity they need at various times  
> **(2)** Bids from generators specifying how much electricity they can produce at those times and how much they will charge for it.  
>
> Operators accept the generators’ bids in order of cost (least expensive first) until they satisfy the LSEs’ total demand. The **price of the last unit of electricity purchased** is then **paid to every supplier whose bid was accepted, regardless of its actual offer**.  
>
> So, for example, suppose that at 9 a.m. on August 15, four plants serving Washington, D.C. can each produce some amount of electricity for, respectively, **$10/unit, $20/unit, $30/unit, and $40/unit**. Suppose that LSEs’ demand at that time and place is met after the operator accepts the three cheapest bids. The first three generators would then all receive **$30/unit**.  
>
> That amount is (think back to Econ 101) the **marginal cost**—i.e., the added cost of meeting another unit of demand—which is the price an efficient market would produce. FERC calls that cost the **locational marginal price (LMP)**.”

## Why Are Auctions Used?
Throughout California, **billions of electrons** move from one location to another every second to power homes, businesses, and critical infrastructure. 

Because **electricity is difficult to store**, a **real-time mechanism** is needed to balance supply and demand—especially in the modern era, where a **lack of energy supply is unacceptable** due to society’s reliance on electricity. 

Given the **continuous and highly variable nature of energy usage**, auctions provide an **efficient** way to ensure that electricity prices reflect **current market conditions**, including:
- Available supply
- Demand fluctuations
- Transmission and distribution constraints

**Auctions create a marketplace for energy buyers and sellers to meet trade value efficiently.**

## SCP Auctions in California
In California, the **California Independent System Operator (CAISO)** manages single-clearing price auctions as follows:

### **Day-Ahead Market (DAM)**
- LSEs specify how much energy they’ll need in **each 6-minute interval** throughout the following day.  
- CAISO synthesizes this data to generate a **bid curve**, which forecasts the expected number of megawatts of electricity that will be consumed.  
- This bid curve represents **expected demand as a function of time** over 6-minute intervals.

### **Real-Time Market (RTM)**
- The **spot market** operates much closer to **actual electricity delivery** and balances discrepancies between day-ahead schedules and actual conditions.  
- It operates **continuously** throughout the day in **15-minute increments** and also uses the SCP mechanism for price formation.

---

## **Research Question**
The **SCP mechanism** is cited as encouraging **efficiency** in the market and benefiting consumers by incentivizing generators to **bid prices with minimal markup**. But is this actually true?

Specifically, we ask:
1. **Does SCP lend itself to market manipulation?**  
   - Can we identify **optimal bidding strategies** that include **spurious high bids** to **inflate profits (and consumer costs)?**  
2. **Does SCP provide lower costs, on average, than pay-as-bid auctions would?**  
3. **Is there an alternative auction model** that would prevent collusion and encourage participants to **bid their true value?**  

---

## **Methods**
1. **Conceptual Model**  

2. **Game-Theoretic Analysis**  
   - Devise a **game-theoretic explanation** to determine the **Nash equilibria** for SCP markets.  
   - Analyze how equilibria **vary with different bid structures**.  

3. **Optimization with Genetic Algorithms**  
   - Use a **genetic algorithm** to **identify optimal bid structures**.  

4. **Numerical Simulations**  
   - Compare SCP with **pay-as-bid** or **reverse second-price** mechanisms.  

5. **Market Manipulation Investigation** *(If Time Permits)*  
   - Once we **identify how the market can be exploited**, search the publicly available data for **evidence** that utilities/CAISO are engaging in these behaviors.  
   - This data is publicly available through the **Open Access Same-Time Information System (OASIS) database**.

---

### **Next Steps**
We will continue to refine our methods, gather data, and simulate market behaviors to assess the potential for market manipulation in California's SCP auctions.

---
