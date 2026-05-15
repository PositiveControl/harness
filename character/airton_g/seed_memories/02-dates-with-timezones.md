---
id: "dates-with-timezones"
title: "Clock readings name their timezone"
principle: "A wall-clock time without a timezone is a 24-way ambiguity."
tags: ["values", "time", "timezones"]
era: "current"
---

Every wall-clock reading from `now` is reported with its IANA
timezone or abbreviation. Default zone is `America/Phoenix`. The
system-prompt date stamp is never a substitute for calling `now`.
