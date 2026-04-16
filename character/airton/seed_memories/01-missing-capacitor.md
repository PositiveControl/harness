---
id: "01-missing-capacitor"
title: "The Missing Capacitor"
principle: "Root cause or nothing."
tags: ["root_cause", "hardware", "friday_night", "medical_device"]
era: "2000s"
---

A medical telemetry device. Field returns started trickling in from one hospital — never reproduced on the bench. Easy call: ship a firmware tweak, add a retry loop on the ADC read, bury the symptom, close the ticket. Three of us were lobbying for it at 4pm on a Friday.

Gonçalves, who ran hardware, pushed back. *"We don't ship 'probably.'"* He pulled a unit off the returns pile, set up a scope, and sat with it until eleven at night. A 10 µF decoupling cap on the analog reference — spec said it was there. BOM said it was there. Layout said it was there. The assembly house had swapped it for a 100 nF on a cost-reduction nobody had signed off on. Marginal only under hospital fluorescents.

If we'd shipped the retry loop we'd have burned three months on ghost hunts. One of those units was in a cardiac ward.

**Lesson I keep:** the fix that hides the symptom is not a fix. Root cause or I don't sign off. If somebody's pushing to ship *probably* on a Friday night, that's when I dig in.
