"""Country / region gazetteer (harness-lyyr).

Used by ScopeViolationHook to verify that a country named in a reply
actually belongs to the region the user asked about. The data is
deliberately static and offline — no external lookups, no network
dependency. Coverage focuses on the regions a user is likely to name
in a 'which X country' question; obscure microstates may not appear.

Designed as a catcher-internal module today; harness-e83u will lift
the same data into src/harness/tools/geography.py so the model can
verify scope proactively before answering, sharing this file as the
source of truth.

Coverage:
  - South America, North America, Central America, Caribbean
  - Latin America (= Mexico + Central America + South America + Caribbean)
  - Africa (UN regions consolidated)
  - Asia
  - Europe
  - Oceania
  - Middle East (the West Asia subregion, commonly named)
  - Scandinavia / Nordics
  - Sub-Saharan Africa
"""

from __future__ import annotations

# Each subregion is a frozenset of canonical country names + common
# short forms (USA, UK, etc.). The matcher is case-insensitive and
# does a substring search on the reply.


SOUTH_AMERICA: frozenset[str] = frozenset(
    {
        "Argentina",
        "Bolivia",
        "Brazil",
        "Chile",
        "Colombia",
        "Ecuador",
        "Falkland Islands",
        "French Guiana",
        "Guyana",
        "Paraguay",
        "Peru",
        "Suriname",
        "Uruguay",
        "Venezuela",
    }
)


# UN M49 "Northern America" — the US + Canada + their immediate
# neighbours. Excludes Mexico (lives in Central America / Latin
# America by UN M49). Common usage often groups Mexico with North
# America too; harness-lyyr stays with UN M49 for determinism.
NORTH_AMERICA: frozenset[str] = frozenset(
    {
        "Bermuda",
        "Canada",
        "Greenland",
        "Saint Pierre and Miquelon",
        "United States",
        "United States of America",
        "USA",
        "US",
        "U.S.",
        "U.S.A.",
    }
)


CENTRAL_AMERICA: frozenset[str] = frozenset(
    {
        "Belize",
        "Costa Rica",
        "El Salvador",
        "Guatemala",
        "Honduras",
        "Nicaragua",
        "Panama",
    }
)


CARIBBEAN: frozenset[str] = frozenset(
    {
        "Anguilla",
        "Antigua and Barbuda",
        "Aruba",
        "Bahamas",
        "Barbados",
        "British Virgin Islands",
        "Cayman Islands",
        "Cuba",
        "Curaçao",
        "Dominica",
        "Dominican Republic",
        "Grenada",
        "Guadeloupe",
        "Haiti",
        "Jamaica",
        "Martinique",
        "Montserrat",
        "Puerto Rico",
        "Saint Barthélemy",
        "Saint Kitts and Nevis",
        "Saint Lucia",
        "Saint Martin",
        "Saint Vincent and the Grenadines",
        "Sint Maarten",
        "Trinidad and Tobago",
        "Turks and Caicos Islands",
    }
)


# Latin America (regional grouping) = Mexico + Central America + South
# America + the Hispanophone Caribbean. The harness uses the broader
# UN M49 'Latin America and the Caribbean' union for simplicity.
LATIN_AMERICA: frozenset[str] = frozenset({"Mexico"} | CENTRAL_AMERICA | SOUTH_AMERICA | CARIBBEAN)


AFRICA: frozenset[str] = frozenset(
    {
        "Algeria",
        "Angola",
        "Benin",
        "Botswana",
        "Burkina Faso",
        "Burundi",
        "Cabo Verde",
        "Cape Verde",
        "Cameroon",
        "Central African Republic",
        "Chad",
        "Comoros",
        "Congo",
        "Democratic Republic of the Congo",
        "DRC",
        "Djibouti",
        "Egypt",
        "Equatorial Guinea",
        "Eritrea",
        "Eswatini",
        "Swaziland",
        "Ethiopia",
        "Gabon",
        "Gambia",
        "Ghana",
        "Guinea",
        "Guinea-Bissau",
        "Ivory Coast",
        "Côte d'Ivoire",
        "Kenya",
        "Lesotho",
        "Liberia",
        "Libya",
        "Madagascar",
        "Malawi",
        "Mali",
        "Mauritania",
        "Mauritius",
        "Morocco",
        "Mozambique",
        "Namibia",
        "Niger",
        "Nigeria",
        "Rwanda",
        "Sao Tome and Principe",
        "São Tomé and Príncipe",
        "Senegal",
        "Seychelles",
        "Sierra Leone",
        "Somalia",
        "South Africa",
        "South Sudan",
        "Sudan",
        "Tanzania",
        "Togo",
        "Tunisia",
        "Uganda",
        "Western Sahara",
        "Zambia",
        "Zimbabwe",
    }
)


# Sub-Saharan Africa = Africa minus North African (Mediterranean)
# countries. Algeria / Egypt / Libya / Morocco / Tunisia / Western
# Sahara / Sudan are typically excluded.
SUB_SAHARAN_AFRICA: frozenset[str] = AFRICA - frozenset(
    {
        "Algeria",
        "Egypt",
        "Libya",
        "Morocco",
        "Sudan",
        "Tunisia",
        "Western Sahara",
    }
)


# UN M49 'North Africa' — Mediterranean Africa.
NORTH_AFRICA: frozenset[str] = frozenset(
    {
        "Algeria",
        "Egypt",
        "Libya",
        "Morocco",
        "Sudan",
        "Tunisia",
        "Western Sahara",
    }
)


ASIA: frozenset[str] = frozenset(
    {
        "Afghanistan",
        "Armenia",
        "Azerbaijan",
        "Bahrain",
        "Bangladesh",
        "Bhutan",
        "Brunei",
        "Cambodia",
        "China",
        "Cyprus",
        "Georgia",
        "India",
        "Indonesia",
        "Iran",
        "Iraq",
        "Israel",
        "Japan",
        "Jordan",
        "Kazakhstan",
        "Kuwait",
        "Kyrgyzstan",
        "Laos",
        "Lebanon",
        "Malaysia",
        "Maldives",
        "Mongolia",
        "Myanmar",
        "Burma",
        "Nepal",
        "North Korea",
        "Oman",
        "Pakistan",
        "Palestine",
        "Philippines",
        "Qatar",
        "Saudi Arabia",
        "Singapore",
        "South Korea",
        "Sri Lanka",
        "Syria",
        "Taiwan",
        "Tajikistan",
        "Thailand",
        "Timor-Leste",
        "East Timor",
        "Turkey",
        "Turkmenistan",
        "UAE",
        "United Arab Emirates",
        "Uzbekistan",
        "Vietnam",
        "Yemen",
    }
)


# Middle East / West Asia (commonly named distinctly from broader Asia).
MIDDLE_EAST: frozenset[str] = frozenset(
    {
        "Bahrain",
        "Cyprus",
        "Egypt",
        "Iran",
        "Iraq",
        "Israel",
        "Jordan",
        "Kuwait",
        "Lebanon",
        "Oman",
        "Palestine",
        "Qatar",
        "Saudi Arabia",
        "Syria",
        "Turkey",
        "UAE",
        "United Arab Emirates",
        "Yemen",
    }
)


EUROPE: frozenset[str] = frozenset(
    {
        "Albania",
        "Andorra",
        "Austria",
        "Belarus",
        "Belgium",
        "Bosnia and Herzegovina",
        "Bulgaria",
        "Croatia",
        "Cyprus",
        "Czech Republic",
        "Czechia",
        "Denmark",
        "Estonia",
        "Finland",
        "France",
        "Germany",
        "Greece",
        "Hungary",
        "Iceland",
        "Ireland",
        "Italy",
        "Kosovo",
        "Latvia",
        "Liechtenstein",
        "Lithuania",
        "Luxembourg",
        "Malta",
        "Moldova",
        "Monaco",
        "Montenegro",
        "Netherlands",
        "North Macedonia",
        "Norway",
        "Poland",
        "Portugal",
        "Romania",
        "Russia",
        "San Marino",
        "Serbia",
        "Slovakia",
        "Slovenia",
        "Spain",
        "Sweden",
        "Switzerland",
        "Ukraine",
        "United Kingdom",
        "UK",
        "Britain",
        "England",
        "Scotland",
        "Wales",
        "Vatican City",
    }
)


# Nordic / Scandinavian countries.
SCANDINAVIA: frozenset[str] = frozenset(
    {
        "Denmark",
        "Finland",
        "Iceland",
        "Norway",
        "Sweden",
    }
)


OCEANIA: frozenset[str] = frozenset(
    {
        "Australia",
        "Fiji",
        "Kiribati",
        "Marshall Islands",
        "Micronesia",
        "Nauru",
        "New Zealand",
        "Palau",
        "Papua New Guinea",
        "Samoa",
        "Solomon Islands",
        "Tonga",
        "Tuvalu",
        "Vanuatu",
    }
)


# REGION_COUNTRIES maps a normalized region phrase to its member set.
# Keys are lowercase so user-message matching is case-insensitive.
# Both adjective ('South American') and noun ('South America') forms
# resolve to the same set.
REGION_COUNTRIES: dict[str, frozenset[str]] = {
    "south america": SOUTH_AMERICA,
    "south american": SOUTH_AMERICA,
    "north america": NORTH_AMERICA,
    "north american": NORTH_AMERICA,
    "central america": CENTRAL_AMERICA,
    "central american": CENTRAL_AMERICA,
    "caribbean": CARIBBEAN,
    "latin america": LATIN_AMERICA,
    "latin american": LATIN_AMERICA,
    "africa": AFRICA,
    "african": AFRICA,
    "sub-saharan africa": SUB_SAHARAN_AFRICA,
    "sub-saharan african": SUB_SAHARAN_AFRICA,
    "subsaharan africa": SUB_SAHARAN_AFRICA,
    "north africa": NORTH_AFRICA,
    "north african": NORTH_AFRICA,
    "asia": ASIA,
    "asian": ASIA,
    "europe": EUROPE,
    "european": EUROPE,
    "scandinavia": SCANDINAVIA,
    "scandinavian": SCANDINAVIA,
    "nordic": SCANDINAVIA,
    "nordics": SCANDINAVIA,
    "oceania": OCEANIA,
    "oceanian": OCEANIA,
    "australian": OCEANIA,
    "middle east": MIDDLE_EAST,
    "middle eastern": MIDDLE_EAST,
}


# Compile-time check: every region key should map to a non-empty set.
# A typo or empty alias would silently disable scope checks for that
# region, so we surface it at import time.
assert all(REGION_COUNTRIES.values()), "every REGION_COUNTRIES entry must be non-empty"


def countries_in_region(region: str) -> frozenset[str]:
    """Return the country set for `region` (case-insensitive lookup).
    Returns an empty frozenset for unknown regions — callers should
    treat that as 'no scope check available' and stay silent."""
    return REGION_COUNTRIES.get(region.strip().lower(), frozenset())


def in_region(country: str, region: str) -> bool:
    """Is `country` a member of `region`? Case-insensitive on both
    sides. Unknown regions return False (the caller should NOT use
    this for affirmation when the region is unknown — use
    countries_in_region(...) explicitly to detect that case)."""
    members = countries_in_region(region)
    if not members:
        return False
    country_lower = country.strip().lower()
    return any(c.lower() == country_lower for c in members)
