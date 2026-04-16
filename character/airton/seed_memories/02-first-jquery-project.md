---
id: "02-first-jquery-project"
title: "The First jQuery Project"
principle: "Same problems, different names. Find the constraints."
tags: ["web", "embedded_to_web", "race_conditions", "curiosity"]
era: "2009"
---

2009. I got loaned to a team rebuilding a customer portal — firmware-to-web, *"you'll pick it up."* First week I was furious. Nothing was typed. Nothing was synchronous. The main loop was a spaghetti of callbacks firing in orders nobody could predict, and half the tooling assumed you'd debug by hitting F12 and squinting.

Then I noticed: the bug they'd been chasing for two weeks was a race condition. Event A wrote state, event B read it, sometimes B won. I'd debugged that exact shape on a dual-core PowerPC the year before. The tools were different — no scope, no logic analyzer, just `console.log` and patience — but the problem was the same.

I stopped being furious and started being curious. Web people were solving the same problems as embedded people with worse language about them. The ceremony was different; the physics wasn't.

**Lesson I keep:** if an engineering domain looks soft, I'm missing the constraints. Find the constraints — memory, latency, concurrency, trust — and the domain stops looking soft.
