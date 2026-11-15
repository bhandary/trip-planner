"""One named test per case in Sunil's probe scripts, plus Shaun's two cases."""

import json

import pytest
from langchain_core.messages import ToolMessage

import agent

NO_FLIGHTS = "No flights found for these dates."
UNAVAILABLE = "The flight source is unavailable."


def _hotels(*names, city="Lisbon"):
    return {
        "city": city,
        "checkin": "2026-12-01",
        "checkout": "2026-12-04",
        "hotels": [
            {"name": name, "nightly": "150 EUR", "link": f"https://skiplagged.com/h/{index}"}
            for index, name in enumerate(names, 1)
        ],
    }


def _flights():
    return {
        "outbound_date": "2026-12-01",
        "return_date": "2026-12-04",
        "itineraries": [
            {
                "price": "300",
                "currency": "EUR",
                "booking_url": "https://kiwi.com/b1",
                "legs": [
                    "SFO -> LIS depart 2026-12-01T08:00 arrive 2026-12-02T06:00 TAP TP238",
                    "LIS -> SFO depart 2026-12-04T12:00 arrive 2026-12-04T16:00 TAP TP237",
                ],
                "notes": [],
            },
            {
                "price": "412",
                "currency": "EUR",
                "booking_url": "https://kiwi.com/b2",
                "legs": ["SFO -> LIS depart 2026-12-01T19:00 arrive 2026-12-02T15:00 UA UA90"],
                "notes": [],
            },
        ],
    }


def _one_flight(price="300"):
    record = _flights()
    record["itineraries"] = [dict(record["itineraries"][0], price=price)]
    record["itineraries"][0]["legs"] = [record["itineraries"][0]["legs"][0]]
    return record


def test_probe_hotel_day_plan_and_checkin_stay():
    text = agent.replace_hotel_section(
        "Here is your plan.\n"
        "Hotels in Lisbon:\n"
        "1. Memmo Alfama - nice\n"
        "Itinerary:\n"
        "- Day 1: Land, check in at Memmo Alfama, walk to the river.\n"
        "- Day 2: Tram 28 through Lisboa old town.\n"
        "Hotel check-in is at 3 PM.\n",
        _hotels("Memmo Alfama", "Lisboa"),
    )
    assert "1. Memmo Alfama - nice" not in text
    assert "- Day 1: Land, check in at Memmo Alfama, walk to the river." in text
    assert "- Day 2: Tram 28 through Lisboa old town." in text
    assert "Hotel check-in is at 3 PM." in text


def test_probe_priced_flight_line_drops_and_day_plan_stays():
    text = agent.replace_flight_section(
        "Flights:\n"
        "1. TP238 300 EUR\n"
        "Itinerary:\n"
        "- Day 1: Fly TP238 overnight, arrive 6 AM.\n"
        "- Day 2: Explore Belem.\n",
        _one_flight(),
    )
    assert "1. TP238 300 EUR" not in text
    assert "- Day 1: Fly TP238 overnight, arrive 6 AM." in text
    assert "- Day 2: Explore Belem." in text


def test_probe_digits_only_flight_number_dedupes():
    items = [
        {"price": "300", "legs": ["SFO -> LIS depart 2026-12-01 08:00 arrive 2026-12-02 06:00 TAP 238"]},
        {"price": "250", "legs": ["SFO -> LIS depart 2026-12-01 08:00 arrive 2026-12-02 06:00 TAP 238"]},
    ]
    kept = agent.dedupe_itineraries_by_legs(items)
    assert len(kept) == 1
    assert kept[0]["price"] == "250"


def test_probe_different_departure_times_stay_separate():
    items = [
        {"price": "300", "legs": ["SFO -> LIS depart 2026-12-01 08:00 arrive 2026-12-02 06:00 TP238"]},
        {"price": "250", "legs": ["SFO -> LIS depart 2026-12-01 21:00 arrive 2026-12-02 19:00 TP238"]},
    ]
    assert len(agent.dedupe_itineraries_by_legs(items)) == 2


def test_probe_blank_and_unparsed_fares_do_not_raise():
    assert agent._fare_amount("") == float("inf")
    assert agent._fare_amount(None) == float("inf")
    assert agent._fare_amount("abc") == float("inf")
    assert agent._fare_amount("1,234.50") == 1234.50
    assert agent._fare_amount("300") == 300


def test_probe_duplicate_hotel_list_is_replaced():
    text = agent.replace_hotel_section(
        "Plan:\n"
        "Hotels in Lisbon (Dec 1-4):\n"
        "- Memmo Alfama | 150 EUR per night\n"
        "- Lisboa | 90 EUR per night\n"
        "Top picks:\n"
        "1. **Memmo Alfama** - boutique, river views\n"
        "2. Lisboa - budget\n"
        "Enjoy!",
        _hotels("Memmo Alfama", "Lisboa"),
    )
    lines = text.splitlines()
    assert "- Memmo Alfama | 150 EUR per night" not in lines
    assert "- Lisboa | 90 EUR per night" not in lines
    assert "1. **Memmo Alfama**" not in text
    assert "2. Lisboa - budget" not in text
    assert "Enjoy!" in text
    assert text.count("Memmo Alfama") == 1
    assert text.count("Lisboa") == 1


def test_probe_heading_with_a_period_replaces_the_old_list():
    text = agent.replace_hotel_section(
        "### Hotels in Washington, D.C.\n"
        "- Old Hotel A - 200 USD\n"
        "- Old Hotel B - 180 USD\n"
        "\n"
        "Itinerary:\n"
        "- Day 1: arrive",
        _hotels("Hotel New", city="Washington"),
    )
    assert "### Hotels in Washington, D.C." not in text
    assert "Old Hotel A" not in text
    assert "Old Hotel B" not in text
    assert "- Day 1: arrive" in text
    assert "Hotel New" in text


def test_probe_duplicate_flight_lines_drop_and_day_line_stays():
    text = agent.replace_flight_section(
        "Here you go.\n"
        "Flights (Dec 1-4):\n"
        "- 300 EUR | https://kiwi.com/b1\n"
        "Best option:\n"
        "- TP238 SFO-LIS for 300 EUR, book at https://kiwi.com/b1\n"
        "- TP238 for €300\n"
        "Day plan:\n"
        "- Day 1: Fly TP238 overnight, arrive 6 AM.",
        _one_flight(),
    )
    assert "https://kiwi.com/b1" in text
    assert text.count("https://kiwi.com/b1") == 1
    assert "- TP238 for €300" not in text
    assert "- Day 1: Fly TP238 overnight, arrive 6 AM." in text


@pytest.mark.parametrize(
    ("label", "items", "want"),
    [
        (
            "same-flight-space-time-different-clocks",
            [
                {"price": "300", "legs": ["SFO -> LIS depart 2026-12-01 08:00 arrive 2026-12-02 06:00 TP238"]},
                {"price": "250", "legs": ["SFO -> LIS depart 2026-12-01 21:00 arrive 2026-12-02 19:00 TP238"]},
            ],
            2,
        ),
        (
            "exact-dup-space-separated-time",
            [
                {"price": "300", "legs": ["SFO -> LIS depart 2026-12-01 08:00 arrive 2026-12-02 06:00 TP238"]},
                {"price": "250", "legs": ["SFO -> LIS depart 2026-12-01 08:00 arrive 2026-12-02 06:00 TP238"]},
            ],
            1,
        ),
        (
            "u2-8123-dup",
            [
                {"price": "300", "legs": ["A -> B depart 2026-12-01T08:00 arrive 2026-12-01T10:00 easyJet U2 8123"]},
                {"price": "250", "legs": ["A -> B depart 2026-12-01T08:00 arrive 2026-12-01T10:00 easyJet U2 8123"]},
            ],
            1,
        ),
        (
            "tp-238-dup",
            [
                {"price": "300", "legs": ["A -> B depart 2026-12-01T08:00 arrive 2026-12-01T10:00 TAP TP 238"]},
                {"price": "250", "legs": ["A -> B depart 2026-12-01T08:00 arrive 2026-12-01T10:00 TAP TP 238"]},
            ],
            1,
        ),
        (
            "iso-seconds-dup",
            [
                {"price": "300", "legs": ["A -> B depart 2027-03-15T08:00:00 arrive 2027-03-15T10:00:00 TP238"]},
                {"price": "250", "legs": ["A -> B depart 2027-03-15T08:00:00 arrive 2027-03-15T10:00:00 TP238"]},
            ],
            1,
        ),
        (
            "iso-zulu-millis-dup",
            [
                {"price": "300", "legs": ["A -> B depart 2027-03-15T08:00:00.000Z arrive 2027-03-15T10:00:00.000Z TP238"]},
                {"price": "250", "legs": ["A -> B depart 2027-03-15T08:00:00.000Z arrive 2027-03-15T10:00:00.000Z TP238"]},
            ],
            1,
        ),
    ],
    ids=[
        "same-flight-space-time-different-clocks",
        "exact-dup-space-separated-time",
        "u2-8123-dup",
        "tp-238-dup",
        "iso-seconds-dup",
        "iso-zulu-millis-dup",
    ],
)
def test_probe_leg_dedupe(label, items, want):
    assert len(agent.dedupe_itineraries_by_legs(items)) == want


def test_probe_dict_carrier_and_spaced_number_dedupe():
    items = [
        {
            "price": "3",
            "legs": [{"flightNumber": "238", "carrierCode": "TP", "departureTime": "x1", "arrivalTime": "y1"}],
        },
        {
            "price": "2",
            "legs": [{"flightNumber": "TP 238", "departureTime": "x1", "arrivalTime": "y1"}],
        },
    ]
    kept = agent.dedupe_itineraries_by_legs(items)
    assert len(kept) == 1
    assert kept[0]["price"] == "2"


def test_probe_recovered_flights_sort_by_total():
    ranked = sorted(
        [{"price": "100", "return_price": "500"}, {"price": "300", "return_price": ""}],
        key=agent._itinerary_total,
    )
    assert [item["price"] for item in ranked] == ["300", "100"]


def test_probe_lastminute_is_not_a_configured_target():
    assert "HOTELS_ALT_URL" not in dir(agent)
    assert "lastminute" not in agent.mcp_targets()


@pytest.mark.parametrize(
    "heading",
    [
        "**Hotels in St. Augustine**",
        "Hotels in St. Augustine:",
        "## Hotels (avg. 150 USD/night)",
        "Hotels near the U.S. Capitol",
    ],
)
def test_probe_abbreviation_heading_replaces_the_old_hotel_list(heading):
    assert agent._section_kind(heading) == "hotels"
    text = agent.replace_hotel_section(
        f"{heading}\n- Old Hotel A - 200 USD\n\nDay 1: walk",
        _hotels("Hotel New", city="St. Augustine"),
    )
    assert heading not in text
    assert "Old Hotel A" not in text
    assert "Day 1: walk" in text


def test_probe_car_section_does_not_embed_the_age_note():
    text = agent.render_car_section(
        {"pickup": "Lisbon Airport", "offers": [{"name": "Fiat", "price": "40", "currency": "EUR"}]}
    )
    assert "driver age" not in text.lower()
    assert "assumed" not in text.lower()


def test_probe_empty_search_drops_an_invented_priced_flight():
    text = agent.apply_coordinator_checks(
        "Plan for Lisbon.\nFlights (Dec 1-4):\n- TP238 SFO-LIS 300 EUR https://www.kiwi.com/deep?x=1\n\nEnjoy.",
        [ToolMessage(NO_FLIGHTS, name="search_flights", tool_call_id="c1")],
    )
    assert NO_FLIGHTS in text
    assert "https://www.kiwi.com/deep?x=1" not in text
    assert "Enjoy." in text


def test_probe_retry_with_an_unstructured_fare_is_not_no_flights():
    text = agent.apply_coordinator_checks(
        "Plan for Lisbon.\nFlights (Dec 1-4):\n- TP238 SFO-LIS 300 EUR https://www.kiwi.com/deep?x=1\n\nEnjoy.",
        [
            ToolMessage(NO_FLIGHTS, name="search_flights", tool_call_id="c1"),
            ToolMessage(
                "Best option: TAP TP238 SFO->LIS 300 EUR https://www.kiwi.com/deep?x=1",
                name="search_flights",
                tool_call_id="c2",
            ),
        ],
    )
    assert NO_FLIGHTS not in text
    assert "https://www.kiwi.com/deep?x=1" in text
    assert agent.active_flight_record(
        [
            ToolMessage(NO_FLIGHTS, name="search_flights", tool_call_id="c1"),
            ToolMessage(
                "Best option: TAP TP238 SFO->LIS 300 EUR https://www.kiwi.com/deep?x=1",
                name="search_flights",
                tool_call_id="c2",
            ),
        ]
    ) is None


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"results": None, "error": "rate limited"}, UNAVAILABLE),
        ({"itineraries": [], "error": "upstream timeout"}, UNAVAILABLE),
        ({"itineraries": [], "message": "No results for these dates"}, NO_FLIGHTS),
        ({"itineraries": []}, NO_FLIGHTS),
    ],
    ids=["rate-limited", "upstream-timeout", "no-results-message", "empty-itineraries"],
)
def test_probe_kiwi_render_empty_or_outage(payload, expected):
    assert agent.render_kiwi_itineraries(payload) == expected


def test_probe_hotel_table_is_sealed():
    table = "| Hotel | Price per night | Link |\n|---|---|---|\n| Memmo Alfama | 150 EUR | https://skiplagged.com/h/1 |\n"
    sealed = agent.seal_hotel_tool_result(table, {"city": "Lisbon"}, tool_name="sk_hotels_search")
    assert sealed != agent._payload_text(table)
    assert "STRUCTURED_HOTELS" in sealed


def test_probe_flight_table_via_hotel_tool_is_not_sealed():
    table = "| Airline | Flight | Stops | Price |\n|---|---|---|---|\n| TAP | TP238 | 0 | 300 |\n"
    sealed = agent.seal_hotel_tool_result(table, {"city": "Lisbon"}, tool_name="sk_hotels_search")
    assert sealed == agent._payload_text(table)


def test_probe_hotel_table_via_flight_tool_is_not_sealed():
    table = "| Hotel | Price per night | Link |\n|---|---|---|\n| Memmo Alfama | 150 EUR | https://skiplagged.com/h/1 |\n"
    sealed = agent.seal_hotel_tool_result(table, {"city": "Lisbon"}, tool_name="sk_flights_search")
    assert sealed == agent._payload_text(table)


def test_probe_nonstop_airport_shuttle_header_is_sealed_as_hotels():
    table = (
        "| Name | Rate | Nonstop airport shuttle |\n"
        "|---|---|---|\n"
        "| Memmo Alfama | 150 EUR | yes |\n"
    )
    sealed = agent.seal_hotel_tool_result(table, {"city": "Lisbon"}, tool_name="sk_hotels_search")
    assert sealed != agent._payload_text(table)
    assert "Memmo Alfama" in sealed


@pytest.mark.parametrize(
    ("line", "kind"),
    [
        ("Hotel check-in is at 3 PM.", None),
        ("**Hotel check-in is at 3 PM.**", None),
        ("Hotels in Lisbon.", None),
        ("Flights from Ft. Lauderdale", "flights"),
        ("Hotels near Mt. Fuji", "hotels"),
        ("Hotels, approx. 150 USD a night", None),
        ("**Hotels in St. Augustine**", "hotels"),
        ("**Hotel Two**", None),
    ],
    ids=[
        "check-in-sentence",
        "bold-check-in-sentence",
        "hotels-in-lisbon-sentence",
        "flights-from-ft-lauderdale",
        "hotels-near-mt-fuji",
        "hotels-approx-sentence",
        "bold-st-augustine",
        "bold-hotel-two",
    ],
)
def test_probe_section_kind(line, kind):
    assert agent._section_kind(line) == kind


def test_probe_bold_checkin_survives_hotel_replace():
    text = agent.replace_hotel_section(
        "Hotels in Lisbon:\n- Memmo Alfama\n\n**Hotel check-in is at 3 PM.**\nEnjoy.",
        _hotels("Memmo Alfama"),
    )
    assert "- Memmo Alfama\n" not in text
    assert "**Hotel check-in is at 3 PM.**" in text
    assert "Enjoy." in text


def test_probe_booking_line_repeats_only_the_rendered_url():
    text = agent.replace_hotel_section(
        "Hotels in Lisbon:\n"
        "- Memmo Alfama\n"
        "\n"
        "Memmo Alfama is lovely.\n"
        "Booking: https://skiplagged.com/h/1\n"
        "Booking: https://other.example/deal\n"
        "Enjoy.",
        _hotels("Memmo Alfama"),
    )
    assert "Memmo Alfama is lovely." in text
    assert "https://skiplagged.com/h/1" in text
    assert text.count("https://skiplagged.com/h/1") == 1
    assert "https://other.example/deal" in text
    assert "Enjoy." in text


def test_probe_repeated_price_fields_drop_and_train_return_stays():
    text = agent.replace_flight_section(
        "Trip plan.\n"
        "Flights (Dec 1-4):\n"
        "- TAP nonstop, great value\n"
        "- UA90 option, 412 EUR\n"
        "- Day 1: Fly TP238 overnight, arrive 6 AM.\n"
        "Price: 300 EUR\n"
        "Outbound: TP238 Dec 1 08:00\n"
        "Return: TP237 Dec 4 12:00\n"
        "Booking: https://kiwi.com/b1\n"
        "\n"
        "Itinerary:\n"
        "- Day 2: Explore Belem.\n"
        "- Day 4: Fly home on TP237 (lands 4 PM).\n"
        "Return: Dec 30 by train to Porto, 30 EUR.\n"
        "Tips:\n"
        "- Pack layers.",
        _flights(),
    )
    assert "TAP nonstop, great value" not in text
    assert "UA90 option, 412 EUR" not in text
    assert "Price: 300 EUR" not in text
    assert "Outbound: TP238 Dec 1 08:00" not in text
    assert "Return: TP237 Dec 4 12:00" not in text
    assert text.count("https://kiwi.com/b1") == 1
    assert "- Day 1: Fly TP238 overnight, arrive 6 AM." in text
    assert "- Day 2: Explore Belem." in text
    assert "- Day 4: Fly home on TP237 (lands 4 PM)." in text
    assert "Return: Dec 30 by train to Porto, 30 EUR." in text
    assert "- Pack layers." in text


def test_probe_blank_itinerary_price_does_not_crash_on_a_price_line():
    record = _one_flight()
    record["itineraries"][0]["price"] = ""
    text = agent.replace_flight_section("Flights:\n- x\nPrice: TBD\nEnjoy.", record)
    assert "Price: TBD" in text
    assert "Enjoy." in text


def test_probe_dedupe_still_keeps_the_cheaper_copy():
    items = [
        {"price": "300", "legs": ["A -> B depart 2026-12-01T08:00 arrive 2026-12-01T10:00 TP238"]},
        {"price": "250", "legs": ["A -> B depart 2026-12-01T08:00 arrive 2026-12-01T10:00 TP238"]},
    ]
    assert [item["price"] for item in agent.dedupe_itineraries_by_legs(items)] == ["250"]


def test_probe_empty_search_keeps_alternate_and_day_line():
    text = agent.apply_coordinator_checks(
        "Plan for Lisbon.\n"
        "Flights (Dec 1-4):\n"
        "- TP238 SFO-LIS 300 EUR https://www.kiwi.com/deep?x=1\n"
        "- Alternate (different dates): TP240 on Dec 3, 280 EUR\n"
        "- Day 1: arrive and rest\n"
        "\n"
        "Enjoy.",
        [ToolMessage(NO_FLIGHTS, name="search_flights", tool_call_id="c1")],
    )
    assert NO_FLIGHTS in text
    assert "https://www.kiwi.com/deep?x=1" not in text
    assert "- Alternate (different dates): TP240 on Dec 3, 280 EUR" in text
    assert "- Day 1: arrive and rest" in text
    assert "Enjoy." in text


def test_probe_earlier_fare_then_empty_latest_is_not_no_flights():
    text = agent.apply_coordinator_checks(
        "Plan for Lisbon.\nFlights (Dec 1-4):\n- TP238 SFO-LIS 300 EUR https://www.kiwi.com/deep?x=1\n\nEnjoy.",
        [
            ToolMessage(
                "Best option: TAP TP238 SFO->LIS 300 EUR https://www.kiwi.com/deep?x=1",
                name="search_flights",
                tool_call_id="c1",
            ),
            ToolMessage(NO_FLIGHTS, name="search_flights", tool_call_id="c2"),
        ],
    )
    assert NO_FLIGHTS not in text
    assert "https://www.kiwi.com/deep?x=1" in text


def test_probe_error_payload_is_not_an_empty_search():
    text = agent.apply_coordinator_checks(
        "Plan for Lisbon.\nFlights (Dec 1-4):\n- TP238 SFO-LIS 300 EUR https://www.kiwi.com/deep?x=1\n\nEnjoy.",
        [ToolMessage(json.dumps({"itineraries": [], "error": "timeout"}), name="search_flights", tool_call_id="c1")],
    )
    assert NO_FLIGHTS not in text
    assert "Enjoy." in text


def test_probe_passport_and_metro_tips_stay_after_a_blank():
    text = agent.replace_flight_section(
        "Flights:\n- old TP238 line\n\n- Bring a passport valid 6 months\n- Lisbon airport metro is 1.80 EUR\nEnjoy.",
        _one_flight(),
    )
    assert "old TP238 line" not in text
    assert "- Bring a passport valid 6 months" in text
    assert "- Lisbon airport metro is 1.80 EUR" in text
    assert "Enjoy." in text


def test_probe7_structured_bold_hotel_blocks_drop_and_unlisted_block_stays():
    text = agent.replace_hotel_section(
        "Here's your plan.\n"
        "\n"
        "### Hotels in Lisbon\n"
        "\n"
        "**Memmo Alfama**\n"
        "- Nightly rate: 150 EUR\n"
        "- Booking: https://skiplagged.com/h/1\n"
        "\n"
        "**Inn Rossio Hotel**\n"
        "- Nightly rate: 67 EUR\n"
        "\n"
        "**Hotel Avenida Palace**\n"
        "- Nightly rate: 240 EUR\n"
        "\n"
        "### Day plan\n"
        "- Day 1: arrive",
        _hotels("Memmo Alfama", "Inn Rossio Hotel"),
    )
    assert text.count("**Memmo Alfama**") == 0
    assert "- Nightly rate: 150 EUR" not in text
    assert "https://skiplagged.com/h/1" in text
    assert text.count("https://skiplagged.com/h/1") == 1
    assert "**Inn Rossio Hotel**" not in text
    assert "- Nightly rate: 67 EUR" not in text
    assert "**Hotel Avenida Palace**" in text
    assert "- Nightly rate: 240 EUR" in text
    assert "- Day 1: arrive" in text


def test_probe7_bullets_between_blanks_drop_only_structured_hotels():
    text = agent.replace_hotel_section(
        "Hotels in Lisbon:\n"
        "- Memmo Alfama - 150 EUR\n"
        "\n"
        "- Inn Rossio Hotel - 67 EUR\n"
        "\n"
        "- Hotel Avenida Palace - 240 EUR\n"
        "\n"
        "Enjoy.",
        _hotels("Memmo Alfama", "Inn Rossio Hotel"),
    )
    assert "- Memmo Alfama - 150 EUR" not in text
    assert "- Inn Rossio Hotel - 67 EUR" not in text
    assert "- Hotel Avenida Palace - 240 EUR" in text
    assert "Enjoy." in text


def test_probe7_numbered_flight_offers_drop():
    text = agent.replace_flight_section(
        "Flights (Dec 1-4):\n"
        "1. TAP TP238 nonstop - 300 EUR\n"
        "\n"
        "2. United UA90 - 412 EUR\n"
        "\n"
        "3. Delta DL100 - 380 EUR\n"
        "\n"
        "Day plan:\n"
        "- Day 1: arrive",
        _flights(),
    )
    assert "1. TAP TP238 nonstop - 300 EUR" not in text
    assert "2. United UA90 - 412 EUR" not in text
    assert "3. Delta DL100 - 380 EUR" not in text
    assert "- Day 1: arrive" in text


def test_probe7_bold_flight_option_blocks_drop():
    text = agent.replace_flight_section(
        "Flights (Dec 1-4):\n"
        "**Option 1: TAP**\n"
        "- Fare: 300 EUR\n"
        "\n"
        "**Option 2: Delta DL100**\n"
        "- Fare: 380 EUR\n"
        "- Book: https://www.kiwi.com/deep?id=zzz\n"
        "\n"
        "Enjoy.",
        _flights(),
    )
    assert "Option 1: TAP" not in text
    assert "Fare: 300 EUR" not in text
    assert "Option 2: Delta DL100" not in text
    assert "Fare: 380 EUR" not in text
    assert "deep?id=zzz" not in text
    assert "Enjoy." in text


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("No results", NO_FLIGHTS),
        ("Service temporarily unavailable", UNAVAILABLE),
        ("Request timed out", UNAVAILABLE),
        ("No flights found; upstream returned 0 rows", NO_FLIGHTS),
        ("Rate limit exceeded", UNAVAILABLE),
    ],
    ids=["no-results", "unavailable", "timed-out", "upstream-zero-rows", "rate-limit"],
)
def test_probe7_kiwi_message(message, expected):
    assert agent.render_kiwi_itineraries({"itineraries": [], "message": message}) == expected


def test_probe8_bold_day_heading_after_flights_stays():
    text = agent.replace_flight_section(
        "Flights (Dec 1-4):\n"
        "- TAP TP238 300 EUR\n"
        "\n"
        "**Day 1 - Arrival**\n"
        "- Morning: check in, nap\n"
        "- Evening: Fado show, 35 EUR\n"
        "\n"
        "Enjoy.",
        _one_flight(),
    )
    assert "TAP TP238 300 EUR" not in text
    assert "**Day 1 - Arrival**" in text
    assert "- Morning: check in, nap" in text
    assert "- Evening: Fado show, 35 EUR" in text
    assert "Enjoy." in text


def test_probe8_ground_transport_after_flights_stays():
    text = agent.replace_flight_section(
        "Flights (Dec 1-4):\n"
        "- TAP TP238 300 EUR\n"
        "\n"
        "**Getting around**\n"
        "- Train IC520 to Porto, 25 EUR\n"
        "- Metro to downtown, 1.80 EUR\n"
        "- Price: 7-day Lisboa Card 50 EUR\n"
        "\n"
        "Enjoy.",
        _one_flight(),
    )
    assert "**Getting around**" in text
    assert "- Train IC520 to Porto, 25 EUR" in text
    assert "- Metro to downtown, 1.80 EUR" in text
    assert "- Price: 7-day Lisboa Card 50 EUR" in text
    assert "Enjoy." in text


def test_probe8_prose_paragraph_after_flights_stays():
    text = agent.replace_flight_section(
        "Flights (Dec 1-4):\n"
        "- TAP TP238 300 EUR\n"
        "\n"
        "The TAP flight is the best value; book soon because fares rise in December.\n"
        "\n"
        "Enjoy.",
        _one_flight(),
    )
    assert "The TAP flight is the best value; book soon because fares rise in December." in text
    assert "Enjoy." in text


def test_probe8_plain_getting_around_after_flights_stays():
    text = agent.replace_flight_section(
        "Flights (Dec 1-4):\n- TAP TP238 300 EUR\n\nGetting around\n- Metro 1.80 EUR\n\nEnjoy.",
        _one_flight(),
    )
    assert "Getting around" in text
    assert "- Metro 1.80 EUR" in text
    assert "Enjoy." in text


def test_probe8_stale_dl100_block_drops_and_tips_stay():
    text = agent.replace_flight_section(
        "Flights (Dec 1-4):\n"
        "**Option 1: TAP**\n"
        "- Fare: 300 EUR\n"
        "\n"
        "**Option 2: Delta DL100**\n"
        "- Fare: 380 EUR\n"
        "\n"
        "- Bring a passport valid 6 months\n"
        "- Lisbon airport metro is 1.80 EUR\n"
        "Enjoy.",
        _one_flight(),
    )
    assert "Option 1: TAP" not in text
    assert "Fare: 300 EUR" not in text
    assert "Option 2: Delta DL100" not in text
    assert "Fare: 380 EUR" not in text
    assert "- Bring a passport valid 6 months" in text
    assert "- Lisbon airport metro is 1.80 EUR" in text
    assert "Enjoy." in text


def test_probe8_bold_day_heading_after_hotels_stays():
    text = agent.replace_hotel_section(
        "Hotels in Lisbon:\n"
        "- Memmo Alfama 150 EUR\n"
        "\n"
        "**Day 1 - Arrival**\n"
        "- Morning: check in\n"
        "- Evening: Fado, 35 EUR\n"
        "\n"
        "Enjoy.",
        _hotels("Memmo Alfama"),
    )
    assert "- Memmo Alfama 150 EUR" not in text
    assert "**Day 1 - Arrival**" in text
    assert "- Morning: check in" in text
    assert "- Evening: Fado, 35 EUR" in text
    assert "Enjoy." in text


def test_probe8_tips_after_the_hotel_list_stay():
    text = agent.replace_hotel_section(
        "Hotels in Lisbon:\n- Memmo Alfama 150 EUR\n\n- Bring a plug adapter\n- City tax is 4 EUR per night\n\nEnjoy.",
        _hotels("Memmo Alfama"),
    )
    assert "- Bring a plug adapter" in text
    assert "- City tax is 4 EUR per night" in text
    assert "Enjoy." in text


def test_probe8_short_hotel_prose_after_the_list_stays():
    text = agent.replace_hotel_section(
        "Hotels in Lisbon:\n- Memmo Alfama 150 EUR\n\nMemmo has a rooftop pool, ask for a river-view room.\n\nEnjoy.",
        _hotels("Memmo Alfama"),
    )
    assert "Memmo has a rooftop pool, ask for a river-view room." in text
    assert "Enjoy." in text


def test_probe8_checkin_stays_when_only_the_structured_hotel_block_drops():
    text = agent.replace_hotel_section(
        "### Hotels in Lisbon\n"
        "\n"
        "**Memmo Alfama**\n"
        "- Nightly rate: 150 EUR\n"
        "\n"
        "**Inn Rossio Hotel**\n"
        "- Nightly rate: 67 EUR\n"
        "\n"
        "**Hotel check-in is at 3 PM.**\n"
        "\n"
        "### Day plan\n"
        "- Day 1: arrive",
        _hotels("Memmo Alfama"),
    )
    assert text.count("Memmo Alfama") == 1
    assert "- Nightly rate: 150 EUR" not in text
    assert "**Inn Rossio Hotel**" in text
    assert "- Nightly rate: 67 EUR" in text
    assert "**Hotel check-in is at 3 PM.**" in text
    assert "- Day 1: arrive" in text


def test_probe_blank_fare_renders_price_unavailable():
    text = agent.render_flight_section(
        {"itineraries": [{"price": "", "currency": "EUR", "booking_url": "https://kiwi.com/b1", "legs": []}]}
    )
    assert "price unavailable" in text
    assert "- EUR" not in text


def test_probe9_fare_amount_does_not_delete_budget_lines():
    text = agent.replace_flight_section(
        "Flights (Dec 1-4):\n"
        "- TAP 150 EUR\n"
        "\n"
        "Budget notes:\n"
        "- City tax 150 EUR for the stay\n"
        "- Food budget about 150 EUR\n"
        "\n"
        "Enjoy.",
        _one_flight("150"),
    )
    assert "- TAP 150 EUR" not in text
    assert "- City tax 150 EUR for the stay" in text
    assert "- Food budget about 150 EUR" in text
    assert "Budget notes:" in text
    assert "Enjoy." in text


def test_probe9_prose_that_names_the_flight_stays():
    text = agent.replace_flight_section(
        "Flights (Dec 1-4):\n"
        "- TAP 150 EUR\n"
        "\n"
        "Your TP238 flight lands at 6 AM, so plan a slow first morning.\n"
        "TP238 is the only nonstop option.\n"
        "\n"
        "Enjoy.",
        _one_flight("150"),
    )
    assert "Your TP238 flight lands at 6 AM, so plan a slow first morning." in text
    assert "TP238 is the only nonstop option." in text
    assert "Enjoy." in text


def test_probe9_day_line_with_the_fare_stays():
    text = agent.replace_flight_section(
        "Flights (Dec 1-4):\n"
        "- TAP 150 EUR\n"
        "\n"
        "Itinerary:\n"
        "- Day 1: Fly TP238 overnight (150 EUR).\n"
        "- Day 3: Sintra tour 150 EUR\n"
        "\n"
        "Enjoy.",
        _one_flight("150"),
    )
    assert "- Day 1: Fly TP238 overnight (150 EUR)." in text
    assert "- Day 3: Sintra tour 150 EUR" in text
    assert "Enjoy." in text


def test_probe9_unlisted_dl100_offer_drops():
    text = agent.replace_flight_section(
        "Flights (Dec 1-4):\n"
        "- TAP 150 EUR\n"
        "\n"
        "**Option 2: Delta DL100**\n"
        "- Fare: 380 EUR\n"
        "\n"
        "- Delta DL100 - 380 EUR https://www.kiwi.com/deep?z=1\n"
        "\n"
        "Enjoy.",
        _one_flight("150"),
    )
    assert "Option 2: Delta DL100" not in text
    assert "Fare: 380 EUR" not in text
    assert "deep?z=1" not in text
    assert "Enjoy." in text


def test_probe9_invented_flight_under_the_heading_drops():
    text = agent.replace_flight_section(
        "Flights (Dec 1-4):\n- TAP TP238 150 EUR\n- Delta DL100 380 EUR\nEnjoy.",
        _one_flight("150"),
    )
    assert "TAP TP238 150 EUR" not in text
    assert "Delta DL100 380 EUR" not in text
    assert "Enjoy." in text


def test_probe9_prose_starting_with_the_hotel_name_stays():
    text = agent.replace_hotel_section(
        "Hotels in Lisbon:\n"
        "- Memmo Alfama 150 EUR\n"
        "\n"
        "Memmo Alfama has a rooftop pool; ask for a river-view room.\n"
        "\n"
        "Enjoy.",
        _hotels("Memmo Alfama", "Lis"),
    )
    assert "Memmo Alfama has a rooftop pool; ask for a river-view room." in text
    assert "Enjoy." in text


def test_probe9_short_hotel_name_does_not_eat_lisbon():
    text = agent.replace_hotel_section(
        "Hotels in Lisbon:\n"
        "- Memmo Alfama 150 EUR\n"
        "\n"
        "Lisbon is hilly, bring good shoes.\n"
        "- Lisbon Card covers trams\n"
        "\n"
        "Enjoy.",
        _hotels("Memmo Alfama", "Lis"),
    )
    assert "Lisbon is hilly, bring good shoes." in text
    assert "- Lisbon Card covers trams" in text
    assert "Enjoy." in text


def test_probe9_day_line_that_names_the_hotel_stays():
    text = agent.replace_hotel_section(
        "Hotels in Lisbon:\n"
        "- Memmo Alfama 150 EUR\n"
        "\n"
        "Itinerary:\n"
        "- Day 1: check in at Memmo Alfama\n"
        "\n"
        "Enjoy.",
        _hotels("Memmo Alfama", "Lis"),
    )
    assert "- Day 1: check in at Memmo Alfama" in text
    assert "Enjoy." in text


def test_probe9_unlisted_hotel_block_after_a_blank_stays():
    text = agent.replace_hotel_section(
        "Hotels in Lisbon:\n"
        "- Memmo Alfama 150 EUR\n"
        "\n"
        "**Hotel Avenida Palace**\n"
        "- Nightly rate: 240 EUR\n"
        "\n"
        "Enjoy.",
        _hotels("Memmo Alfama", "Lis"),
    )
    assert "- Memmo Alfama 150 EUR" not in text
    assert "**Hotel Avenida Palace**" in text
    assert "- Nightly rate: 240 EUR" in text
    assert "Enjoy." in text


def test_shaun_prose_that_starts_with_the_hotel_name_stays():
    text = agent.replace_hotel_section(
        "Hotels in Lisbon:\n- Memmo Alfama 150 EUR\n\nMemmo Alfama is a short walk from the river.\n",
        _hotels("Memmo Alfama"),
    )
    assert "Memmo Alfama is a short walk from the river." in text
    assert "- Memmo Alfama 150 EUR" not in text


def test_shaun_unlisted_bold_hotel_is_not_a_heading_and_keeps_its_rate():
    assert agent._section_kind("**Hotel Two**") is None
    text = agent.replace_hotel_section(
        "### Hotels in Lisbon\n"
        "\n"
        "**Memmo Alfama**\n"
        "- Nightly rate: 150 EUR\n"
        "\n"
        "**Hotel Two**\n"
        "- Nightly rate: 120 EUR\n",
        _hotels("Memmo Alfama"),
    )
    assert "**Hotel Two**" in text
    assert "- Nightly rate: 120 EUR" in text
    assert text.count("Memmo Alfama") == 1
    assert "- Nightly rate: 150 EUR" not in text
