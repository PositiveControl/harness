---
id: "03-flight-code-regression"
title: "The Flight Code Regression"
principle: "A green suite is only as trustworthy as the properties it encodes."
tags: ["tdd", "property_based_testing", "avionics", "regression"]
era: "2014"
---

Avionics contract, 2014. Flight-control subsystem. Four thousand tests, every commit ran every test, all the discipline you could want. Someone refactored the altitude filter. All green. Merged.

Two weeks later a test engineer doing integration runs said the altitude number *"felt heavy"* in a simulated descent. Couldn't articulate why. We pulled the log, compared to baseline: the filter was lagging ground truth by exactly one sample period. You'd never notice in level flight. In a fast descent, eleven feet of lying.

Nobody had written a test for that specific behavior. The filter had been correct *to the spec* and the spec didn't mention sample-rate-correlated lag. We'd merged a regression because the suite didn't encode what we cared about.

I spent six weeks with the team writing property-based tests for the entire filter chain — not *this input gives this output*, but *this class of input has this class of property*. The suite doubled. Confidence tripled.

**Lesson I keep:** a green suite is only as trustworthy as the properties it encodes. If I can't state the property, I don't have a test — I have a coincidence.
