#!/usr/bin/env python3
"""Author a small, sealed-evaluation-independent conversational SFT supplement.

Standard library only. Inputs are this source and the two explicitly permitted
coherence scripts (SYSTEM/schema and training-interface provenance). No public
corpus, tokenizer, evaluation file, model output, network, or GPU is consumed.
"""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import random
import re
from pathlib import Path

VERSION = "bliss-calibration-authored-v1"
SEED = 2026091507
CATEGORIES = (
    "indirect_requests", "practical_choices", "conversational_reference",
    "conversational_repair", "personal_information", "fictional_information",
    "operational_information", "concise_summaries", "format_extract",
    "format_reorder", "format_copy", "format_json", "requested_only",
    "logical_relations", "arithmetic_add_subtract", "arithmetic_groups",
)
# A family denotes a scenario and transformation, not just a prompt wording.
# All validation families are held out; the broad categories intentionally overlap.
FAMILIES = {
    "indirect_requests": ("polite_request_rewrite", "verbose_note_simplify", "message_draft_from_intent"),
    "practical_choices": ("availability_slot", "pickup_rule", "room_constraints"),
    "conversational_reference": ("plural_referent", "same_location_relation", "elliptical_other_person"),
    "conversational_repair": ("destination_correction", "object_clarification", "preference_replacement"),
    "personal_information": ("snack_preference_pair", "weekend_activity_pair", "music_choice_pair"),
    "fictional_information": ("story_captain_pair", "story_book_title_pair", "story_courier_destination_pair"),
    "operational_information": ("parcel_owner_pair", "cabinet_code_pair", "milestone_date_pair"),
    "concise_summaries": ("two_plan_facts", "current_status_from_log", "event_and_stated_reason"),
    "format_extract": ("selected_note_fields", "rows_matching_label", "keep_line_filter"),
    "format_reorder": ("reverse_sequence", "rank_order", "clock_order"),
    "format_copy": ("punctuation_payload", "quoted_spacing_payload", "marked_two_line_payload"),
    "format_json": ("person_room_object", "item_count_object", "task_boolean_object"),
    "requested_only": ("supplied_code_mapping", "explicit_list_membership", "larger_side_label"),
    "logical_relations": ("transitive_order", "all_members_rule", "sufficient_condition"),
    "arithmetic_add_subtract": ("stock_movements", "session_minutes", "voucher_balance"),
    "arithmetic_groups": ("boxes_plus_loose", "division_plus_groups", "trays_minus_removed"),
}
POOL = {
    "train": {
        "names": ["Ada", "Basil", "Celia", "Dev", "Esme", "Flynn", "Greta", "Hugo", "Ines", "Jonah", "Lena", "Milo", "Nell", "Otis", "Pia", "Rosa", "Seth", "Tess", "Uma", "Vera", "Wade", "Xena", "Yara", "Zeke"],
        "places": ["Willow room", "Cedar nook", "Acorn hall", "Beech studio", "Clover loft", "Daisy annex", "Holly cabin", "Juniper office", "Laurel shed", "Moss porch", "Pine cellar", "Thistle booth"],
        "items": ["mugs", "socks", "books", "towels", "crayons", "keys", "lamps", "trays", "scarves", "combs", "cups", "maps"],
        "colors": ["red", "blue", "green", "yellow", "black", "white"],
        "snacks": ["toast", "pretzels", "apples", "crackers", "raisins", "popcorn"],
        "activities": ["drawing", "knitting", "jogging", "reading", "gardening", "baking"],
    },
    "val": {
        "names": ["Alma", "Bruno", "Clio", "Dorian", "Enid", "Faye", "Gus", "Heidi", "Isla", "Jasper", "Keiko", "Lucian"],
        "places": ["Kestrel lodge", "Heron alcove", "Osprey tower", "Finch terrace", "Wren pavilion", "Ibis den", "Lark corner", "Swift gallery", "Dove chamber", "Swan passage", "Robin bay", "Tern hut"],
        "items": ["spoons", "gloves", "brushes", "baskets", "erasers", "coins", "vases", "plates", "belts", "mirrors", "bowls", "charts"],
        "colors": ["teal", "indigo", "violet", "ochre", "coral", "umber"],
        "snacks": ["pears", "oatcakes", "walnuts", "plums", "muffins", "scones"],
        "activities": ["singing", "weaving", "swimming", "dancing", "carving", "sewing"],
    },
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def system_from(path: Path) -> str:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SYSTEM" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Permitted source did not declare SYSTEM")


def authored_example(category, split, variant, index, system, rng):
    p = POOL[split]
    name, other, third = rng.sample(p["names"], 3)
    place, second, third_place = rng.sample(p["places"], 3)
    item, other_item, third_item, fourth_item = rng.sample(p["items"], 4)
    color, other_color = rng.sample(p["colors"], 2)
    snack = rng.choice(p["snacks"])
    activity = rng.choice(p["activities"])
    # Pair index keeps answerable/missing rows in one split and scenario group.
    # Content RNG is reinitialized per pair by the caller for paired categories.
    known = index % 2 == 0
    wording = (index // 2 if category.endswith("information") else index) % 4
    family = FAMILIES[category][variant]
    turns = []
    checks = []

    def turn(user, answer, **check):
        turns.extend([{"role": "user", "content": user}, {"role": "assistant", "content": answer}])
        if check:
            checks.append({"assistant_turn": len(turns) // 2, **check})

    def choose(options):
        return options[wording]

    if category == "indirect_requests":
        if variant == 0:
            raw = choose([f"Give the {item} to {name}.", f"Bring the {item} to the {place}.", f"Put the {item} beside the {other_item}.", f"Ask {name} to bring the {item}."])
            polite = choose([f"Could you please give the {item} to {name}?", f"Could you please bring the {item} to the {place}?", f"Could you please put the {item} beside the {other_item}?", f"Could you please ask {name} to bring the {item}?"])
            user = choose([f'Can you make this request more polite? "{raw}"', f'I need a friendlier version of this: "{raw}"', f'Could you soften the wording? "{raw}"', f'Would you turn this into a courteous request? "{raw}"'])
            turn(user, polite, kind="rewrite", anchors=[name if wording in (0, 3) else (place if wording == 1 else other_item), item])
        elif variant == 1:
            raw = choose([f"The location at which {name} left the {item} is the {place}.", f"It is necessary for {name} to bring the {item}.", f"The {item} belonging to {name} are currently located in the {place}.", f"{name} is the person who will be carrying the {item}."])
            simple = choose([f"{name} left the {item} in the {place}.", f"{name} needs to bring the {item}.", f"{name}'s {item} are in the {place}.", f"{name} will carry the {item}."])
            turn(choose([f'Can you say this more simply? "{raw}"', f'Could you make this easier to read? "{raw}"', f'I need a plain version of this note: "{raw}"', f'Please help me shorten this sentence: "{raw}"']), simple, kind="rewrite", anchors=[name, item])
        else:
            intent = choose([f"tell {name} the {item} are ready for collection at the {place}", f"ask {name} to leave the {item} at the {place}", f"thank {name} for bringing the {item}", f"tell {name} I will take the {item} to the {place}"])
            reply = choose([f"Hi {name}, the {item} are ready to collect at the {place}.", f"Hi {name}, could you leave the {item} at the {place}?", f"Thanks for bringing the {item}, {name}.", f"Hi {name}, I'll take the {item} to the {place}."])
            turn(choose([f"I'm texting someone. Would you help me {intent}? Give me the message.", f"I want to {intent}. What short message could I send?", f"Write the text for me: {intent}.", f"Could you draft a brief message to {intent}?"]), reply, kind="rewrite", anchors=[name, item])
    elif category == "practical_choices":
        if variant == 0:
            hour = rng.randint(8, 15)
            before, fit, after = f"{hour}:00", f"{hour + 1}:00", f"{hour + 2}:00"
            user = choose([f"{name} is free only from {fit} through {fit[:fit.index(':')]}:30. The offered 15-minute appointments start at {before}, {fit}, or {after}. Which should I book?", f"Can you pick a time? The only free window for {name} is {fit}-{fit[:fit.index(':')]}:30. Each appointment lasts 15 minutes; starts available: {before}, {fit}, {after}.", f"I need a 15-minute slot for {name}. Available starts are {before}, {fit}, and {after}; {name} can attend only between {fit} and {fit[:fit.index(':')]}:30. What fits?", f"Booking note: {name}, free {fit}-{fit[:fit.index(':')]}:30; offered starts {before}/{fit}/{after}; duration 15 minutes. Choose a suitable start."])
            turn(user, f"Book {fit}; the 15-minute appointment fits the free window.", kind="choice", selected=fit, allowed=[before, fit, after], evidence=fit)
        elif variant == 1:
            user = choose([f"Pickup instructions: collect the {item} at the {place} if you have a pass; otherwise use the {second}. I have a pass. Where should I go?", f"I don't have a pass. The note says pass holders collect {item} at the {place}; everyone else uses the {second}. Which place applies to me?", f"For the {item}, there are two pickup points: {place} for people with passes, {second} for people without. My pass is in my hand. What is my pickup point?", f"The {item} are available at the {place} with a pass, or the {second} without one. I have no pass; help me choose the right point."])
            dest = place if wording % 2 == 0 else second
            turn(user, f"Go to the {dest}.", kind="choice", selected=dest, allowed=[place, second], evidence=dest)
        else:
            needed = rng.randint(3, 9)
            turn(choose([f"Help choose a room for {needed} people who need quiet. The {place}: {needed - 1} seats, quiet. The {second}: {needed + 2} seats, quiet. The {third_place}: {needed + 4} seats, noisy. Which meets both needs?", f"Our group has {needed} members and needs a quiet space. Options: {place} ({needed - 1} seats, quiet); {second} ({needed} seats, quiet); {third_place} ({needed + 3} seats, noisy). Where can we meet?", f"Space request: at least {needed} seats and quiet. Room list: {place}, quiet, {needed - 1}; {second}, quiet, {needed + 1}; {third_place}, noisy, {needed + 2}. Pick one that satisfies the request.", f"Can you find a suitable room? We require quiet and {needed} seats. Only these are offered: {place} has {needed - 1} seats and is quiet; {second} has {needed} seats and is quiet; {third_place} has {needed + 5} seats and is noisy."]), f"Use the {second}; it is quiet and has enough seats.", kind="choice", selected=second, allowed=[place, second, third_place], evidence=second)
    elif category == "conversational_reference":
        if variant == 0:
            turn(choose([f"The {item} need to go to the {place}. I'm holding them now.", f"I've picked up the {item}. Their assigned spot is the {place}.", f"The destination for these {item} is the {place}. I've got them with me.", f"I'm carrying the {item}; the note puts them in the {place}."]), "Understood.")
            turn(choose(["Where should I put those?", "Where do these go again?", "Remind me where to take them.", "What's their destination?"]), f"Put the {item} in the {place}.", kind="grounded", anchors=[item, place])
        elif variant == 1:
            turn(choose([f"{name}'s {item} are in the {place}. {other}'s {other_item} are kept in the same place.", f"The {place} holds {name}'s {item}. {other} uses that same location for the {other_item}.", f"{name} leaves the {item} in the {place}, and {other} leaves the {other_item} there too.", f"Both {name}'s {item} and {other}'s {other_item} go to the {place}."]), "Got it.")
            turn(choose([f"Where are {other}'s things?", f"And where does {other} keep the {other_item}?", f"Which place holds the {other_item}?", f"Where should I look for {other}'s {other_item}?"]), f"{other}'s {other_item} are in the {place}.", kind="grounded", anchors=[other, other_item, place])
        else:
            turn(choose([f"{name} chose {color}; {other} chose {other_color}. What color did {name} choose?", f"Color choices: {name} = {color}, {other} = {other_color}. Tell me {name}'s choice.", f"In our list, {name} picked {color} and {other} picked {other_color}. Which one did {name} pick?", f"There are two selections: {color} for {name}, {other_color} for {other}. Start with {name}'s color."]), f"{name} chose {color}.")
            turn(choose([f"And {other}?", f"What about {other}?", f"Now {other}'s?", f"How about the choice for {other}?"]), f"{other} chose {other_color}.", kind="grounded", anchors=[other, other_color])
    elif category == "conversational_repair":
        if variant == 0:
            turn(choose([f"Send {name}'s {item} to the {place}.", f"Delivery destination for {name}'s {item}: {place}.", f"The {place} is where {name}'s {item} should go.", f"For now, put the {item} for {name} at the {place}."]), f"The destination is the {place}.")
            turn(choose([f"Actually, make that the {second}. Where should they go?", f"Correction: the destination is the {second}. Confirm the new place.", f"I changed my mind; use the {second}. What's the destination now?", f"Replace that location with the {second}. Where will the {item} go?"]), f"They should go to the {second}.", kind="repair", required=[second], forbidden=[place])
        elif variant == 1:
            turn(choose([f"The {item} are at the {place}; the {other_item} are at the {second}. Where are the {item}?", f"Storage: {item} in the {place}, {other_item} in the {second}. Find the {item}.", f"I put the {item} in the {place} and the {other_item} in the {second}. Remind me where the {item} are.", f"Two locations: {place} for {item}, {second} for {other_item}. I need the {item}; where are they?"]), f"The {item} are at the {place}.")
            turn(choose([f"Sorry, I meant the {other_item}.", f"I asked about the wrong thing. Where are the {other_item}?", f"Let me correct that: I need the {other_item}.", f"The {other_item}, I should have said."]), f"The {other_item} are at the {second}.", kind="repair", required=[other_item, second], forbidden=[place])
        else:
            turn(choose([f"For this order, {name} selected {color}.", f"Keep this order detail: {name}'s chosen color is {color}.", f"The color on {name}'s order is {color}.", f"Order note: {name} wants {color}."]), f"{name}'s selected color is {color}.")
            turn(choose([f"Update from {name}: {other_color} instead. Which color is current?", f"{name} replaced that choice with {other_color}. Tell me the active selection.", f"That choice was cancelled. {name} now wants {other_color}; confirm it.", f"Use {other_color} for {name} from now on. What should the order say?"]), f"{name}'s current choice is {other_color}.", kind="repair", required=[other_color], forbidden=[color])
    elif category in ("personal_information", "fictional_information", "operational_information"):
        if category == "personal_information":
            if variant == 0:
                base, fact, question, answer, missing = f"{name} keeps the {item} in the {place}.", f"{name}'s favorite snack is {snack}.", f"What is {name}'s favorite snack?", f"{name}'s favorite snack is {snack}.", f"The note doesn't say what {name}'s favorite snack is."
            elif variant == 1:
                base, fact, question, answer, missing = f"{name} brought {item} to the {place}.", f"{name} plans to spend Saturday {activity}.", f"What does {name} plan to do on Saturday?", f"{name} plans to spend Saturday {activity}.", f"The note doesn't give {name}'s Saturday plan."
            else:
                music = rng.choice(["jazz", "folk", "blues", "opera", "soul", "ambient music"])
                base, fact, question, answer, missing = f"{name} signed up in the {place}.", f"For this event, {name} chose {music}.", f"Which music did {name} choose for this event?", f"{name} chose {music} for this event.", f"{name}'s music choice isn't included in this note."
        elif category == "fictional_information":
            if variant == 0:
                base, fact, question, answer, missing = f"In this invented story, a ship arrives at the port near the {place} with {item}.", f"Its captain is {name}.", "Who captains the ship in this story?", f"{name} is the ship's captain.", "The story excerpt doesn't identify the ship's captain."
            elif variant == 1:
                title = f"The {color.title()} {item.title()}"
                base, fact, question, answer, missing = f"In an invented story, {name} reads a book at the {place}.", f"The book is called {title}.", "What is the title of the book in this story?", f"The book is called {title}.", "The excerpt doesn't give the book's title."
            else:
                base, fact, question, answer, missing = f"In a made-up tale, {name} is a courier carrying {item}.", f"The courier's destination is the {place}.", "Where is the courier headed in the tale?", f"The courier is headed to the {place}.", "The supplied passage leaves the courier's destination unstated."
        else:
            if variant == 0:
                base, fact, question, answer, missing = f"A parcel containing {item} is at the {place}.", f"The parcel belongs to {name}.", "Who owns the parcel?", f"The parcel belongs to {name}.", "The note doesn't identify the parcel's owner."
            elif variant == 1:
                code = f"T{rng.randint(1000, 9999)}"
                base, fact, question, answer, missing = f"The cabinet in the {place} contains {item}.", f"In this fictional exercise, the cabinet code is {code}.", "What cabinet code does the note give?", f"The cabinet code is {code}.", "No cabinet code is given in the note."
            else:
                date = f"October {rng.randint(1, 28)}"
                base, fact, question, answer, missing = f"{name} is coordinating the {place} project.", f"The milestone is scheduled for {date}.", "What date is the milestone scheduled for?", f"The milestone is scheduled for {date}.", "The milestone date cannot be determined from this brief."
        note = base + (" " + fact if known else "")
        if variant == 2:
            user = choose([f"Here is all the background I have:\n{note}\n{question}", f"Read this short brief: {note}\n{question} Use the brief as your source.", f"A passage says: {note}\nCan you answer this from it: {question}", f"Work from this information alone: {note}\nMy question is: {question}"])
        else:
            user = choose([f"Note: {note}\n{question} Use only the note.", f"Using this note, answer my question. {note}\n{question}", f"What can you tell me from these details? {note}\n{question}", f"I have this information: {note}\nBased on it, {question[0].lower() + question[1:]}"])
        turn(user, answer if known else missing, kind="paired_grounding", answerable=known, base=base, relevant_fact=fact, evidence_present=fact in note, expected_known=answer, expected_missing=missing)
    elif category == "concise_summaries":
        if variant == 0:
            turn(choose([f"Summarize this in one sentence: {name} will bring the {item}. The meeting will be in the {place}.", f"Give me a short recap: our meeting is in the {place}, and {name} is bringing the {item}.", f"Could you combine these plans into one concise sentence? {name}: bring {item}. Meeting location: {place}.", f"Make this note brief: the {place} is the meeting spot; the {item} will be brought by {name}."]), f"{name} will bring the {item} to the meeting in the {place}.", kind="grounded", anchors=[name, item, place])
        elif variant == 1:
            turn(choose([f"Give the current status in one short sentence. Earlier: {name} left the {item} at the {place}. Latest: {other} moved them to the {second}.", f"Summarize where things stand now: {item} delivered by {name} to the {place}; later transferred by {other} to the {second}.", f"Latest-state recap, please. The {item} started at the {place} with {name}. {other} has now taken them to the {second}.", f"Reduce this log to the present location: first {name} stored the {item} at the {place}; then {other} relocated them to the {second}."]), f"The {item} are now at the {second}.", kind="repair", required=[item, second], forbidden=[place])
        else:
            reason = choose(["the first room was occupied", "the original room was being cleaned", "the original room had no chairs", "the first room was locked"])
            turn(choose([f"Please capture the event and reason together: {name}'s group moved to the {place} because {reason}.", f"In a short sentence, what happened and why? {name}'s group switched to the {place}. The stated reason was that {reason}.", f"Condense this update without losing the explanation. Since {reason}, {name}'s group moved to the {place}.", f"Make a brief recap containing the change and its cause: {reason}, so {name}'s group moved to the {place}."]), f"{name}'s group moved to the {place} because {reason}.", kind="grounded", anchors=[name, place, reason])
    elif category == "format_extract":
        if variant == 0:
            turn(choose([f"Extract only the owner and room, joined by ' | '. Note: owner={name}; contents={item}; room={place}.", f"From 'room: {place}; owner: {name}; items: {item}', return owner then room separated by ' | '. No other text.", f"Fields: contents {item}; room {place}; owner {name}. Output just 'owner | room', replacing those labels with their values.", f"Read this entry: {name} is the owner, {place} is the room, {item} are the contents. Give only the owner and room as two values with ' | ' between them."]), f"{name} | {place}", kind="exact", expected=f"{name} | {place}")
        elif variant == 1:
            lines = [f"{name}: collect", f"{other}: wait", f"{third}: collect"]
            rng.shuffle(lines)
            expected = ", ".join(line.split(":")[0] for line in lines if line.endswith(": collect"))
            turn(choose(["Return only the names marked collect, in their listed order, separated by comma-space:\n", "Select the collect names from this list. Keep the original order and output comma-separated names only:\n", "Which names have the label collect? Reply with just those names in list order, using ', ' between them:\n", "Filter out wait entries. Give only the remaining names, comma-space separated, preserving order:\n"]) + "\n".join(lines), expected, kind="extract_rows", rows=lines, tag="collect", separator=", ")
        else:
            lines = [f"keep: {item}", f"skip: {other_item}", f"keep: {third_item}", f"skip: {fourth_item}"]
            rng.shuffle(lines)
            expected = "\n".join(line[6:] for line in lines if line.startswith("keep: "))
            turn(choose(["Copy the values on keep lines; omit the labels. Retain line order, one value per line, with no introduction:\n", "Remove skip lines and strip 'keep: ' from the others. Return the resulting lines alone:\n", "Output only the text following 'keep: ', using one line for each retained value in the same order:\n", "Process these lines: discard skip entries, remove keep prefixes, and preserve the remaining line order. Return only the result:\n"]) + "\n".join(lines), expected, kind="line_filter", lines=lines, prefix="keep: ")
    elif category == "format_reorder":
        if variant == 0:
            seq = [item, other_item, third_item, fourth_item]
            rng.shuffle(seq)
            expected = " > ".join(seq[::-1])
            turn(choose(["Reverse this sequence. Output only the reversed values joined by ' > ': ", "Put the following list in reverse order; use ' > ' between values and nothing else: ", "Read the values from last to first. Return just that sequence, separated by ' > ': ", "Reorder this sequence backwards without changing the values. Give only the result with ' > ' separators: "]) + ", ".join(seq), expected, kind="reverse", values=seq, separator=" > ")
        elif variant == 1:
            ranks = rng.sample(range(1, 31), 3)
            entries = list(zip([name, other, third], ranks))
            expected = "\n".join(n for n, _ in sorted(entries, key=lambda x: x[1]))
            turn(choose(["Sort by rank from lowest to highest. Return names only, one per line:\n", "Put these names in ascending numerical rank order. Output only the names on separate lines:\n", "The smaller rank comes first. Arrange the entries and give just their names, one on each line:\n", "Order the following by increasing rank; omit ranks and commentary from your answer. Use a line per name:\n"]) + "\n".join(f"{n}: rank {r}" for n, r in entries), expected, kind="rank", entries=entries)
        else:
            starts = rng.sample(range(8 * 60, 18 * 60, 5), 3)
            entries = list(zip([place, second, third_place], starts))
            expected = " -> ".join(n for n, _ in sorted(entries, key=lambda x: x[1]))
            turn(choose(["Arrange the stops from earliest to latest, using the 24-hour times. Reply only with their names joined by ' -> ':\n", "Build the visit sequence in time order. Return location names only, separated by ' -> ':\n", "Use these 24-hour appointments to give the chronological route. Format the place names as 'first -> second -> third' without a preface:\n", "Which order do these visits occur in? Return only location names with ' -> ' separators, earliest first:\n"]) + "\n".join(f"{n}: {t // 60:02d}:{t % 60:02d}" for n, t in entries), expected, kind="clock_order", entries=entries)
    elif category == "format_copy":
        code = ("TU" if split == "train" else "VE") + str(rng.randint(100, 999))
        if variant == 0:
            payload = choose([f"{code}: {item}!", f"{name} / {color} / {code}", f"[{code}] {place}.", f"{item.upper()}? {code}."])
            user = choose(["Copy the text after TEXT exactly, with no prefix or quotation marks.\nTEXT\n", "Repeat the following line exactly. Preserve case and punctuation; output the line alone.\n", "Return the exact line below, unchanged and without a code fence:\n", "Please reproduce this line character for character. Your answer must contain only that line:\n"]) + payload
        elif variant == 1:
            payload = choose([f"{code}  {name}", f"{color}   {item}", f"{name}  -  {code}", f"{code}  /  {place}"])
            user = choose(["Copy what is between the quotation marks, excluding the marks. Preserve all spaces. Answer only with the copied text: ", "Reproduce the quoted text exactly, including repeated spaces, but omit the surrounding quotes: ", "Give only the contents of these quotes. Don't change capitalization or spacing: ", "Return this quoted string unchanged without its quote marks or any additional text: "]) + '"' + payload + '"'
        else:
            payload = choose([f"{code}: {name}\n{color} {item}", f"{place}\n({code})", f"{item.upper()}\n{name} / {code}", f"{code} -- {color}\n{place}"])
            user = choose(["Copy the two lines enclosed by BEGIN and END, leaving out the markers. Keep the line break and add nothing:\n", "Reproduce exactly the content inside these markers. Exclude BEGIN and END; preserve the newline:\n", "Return only the two marked lines, with their original case, symbols, and line break:\n", "Transcribe between BEGIN and END. Include the intervening newline, omit marker lines, and use no code fence:\n"]) + "BEGIN\n" + payload + "\nEND"
        turn(user, payload, kind="copy", payload=payload)
    elif category == "format_json":
        if variant == 0:
            expected = {"name": name, "room": place}
            user = choose([f"Return only a JSON object with string keys name and room. Name: {name}. Room: {place}.", f"Convert this to JSON only: name is {name}; room is {place}. Use exactly the keys name and room, with string values.", f"I need a JSON object, no code fence: {{name, room}} using {name} as the name and {place} as the room. Include only those keys.", f"Output exactly two JSON string fields, name and room, using this note: {name} is assigned to the {place}. No prose."])
        elif variant == 1:
            count = rng.randint(0, 65)
            expected = {"item": item, "count": count}
            user = choose([f"JSON only, with exactly item (string) and count (integer): {count} {item}.", f"Encode this stock entry as one JSON object: item={item}; count={count}. Only keys item and count; count must be a number. No other output.", f"Return an object with two fields: item set to {item}, and integer count set to {count}. Use JSON without a code fence.", f"Produce only valid JSON for {item}, quantity {count}. Keys must be item and count; use a string and an integer respectively."])
        else:
            ready = bool(index % 2)
            state = "ready" if ready else "not ready"
            expected = {"task": f"move {item}", "ready": ready}
            user = choose([f"Task: move {item}. Status: {state}. Respond with JSON alone: task as a string and ready as a boolean. Include no other fields.", f"Serialize this entry with precisely two keys, task and ready: 'move {item}' is {state}. Use a JSON boolean for ready and no surrounding explanation.", f"Make a JSON record for the task 'move {item}'. It is {state}. Return only task (string) and ready (true/false).", f"The task called 'move {item}' has status '{state}'. Give its JSON object only, using task for the name and ready for a boolean status."])
        turn(user, dump(expected), kind="json", expected=expected)
    elif category == "requested_only":
        if variant == 0:
            mapping = {name: "A", other: "B", third: "C"}
            selected = rng.choice(list(mapping))
            turn(choose([f"Codes: A={name}, B={other}, C={third}. What is the code for {selected}? Answer with the single letter only.", f"Use this mapping: {name}->A; {other}->B; {third}->C. Return only {selected}'s code.", f"Select a letter for {selected}: A means {name}, B means {other}, C means {third}. Output one letter and no punctuation.", f"Look up {selected} in this key: A: {name}; B: {other}; C: {third}. Give just the matching uppercase letter."]), mapping[selected], kind="mapping", mapping=mapping, selected=selected)
        elif variant == 1:
            values = [item, other_item]
            query = item if known else third_item
            expected = "yes" if query in values else "no"
            turn(choose([f"List: {', '.join(values)}. Is the entry '{query}' in this exact list? Reply only yes or no.", f"Does the list [{', '.join(values)}] include {query}? Lowercase yes or no only, no punctuation.", f"Check membership: {query}; allowed entries: {', '.join(values)}. Output just yes if present, otherwise no.", f"Here are the only listed entries: {', '.join(values)}. Is the entry '{query}' listed? Answer with one lowercase word: yes or no."]), expected, kind="membership", values=values, query=query)
        else:
            left = rng.randint(0, 80)
            right = left if index % 3 == 0 else left + (rng.randint(1, 9) * (1 if index % 3 == 1 else -1))
            expected = "LEFT" if left > right else "RIGHT" if right > left else "TIE"
            turn(choose([f"Left value {left}; right value {right}. Return only LEFT if the left is larger, RIGHT if the right is larger, or TIE if equal.", f"Compare these values: left={left}, right={right}. The only allowed output is LEFT, RIGHT, or TIE, naming the larger side or equality.", f"Choose the larger side for left {left} and right {right}. Give a single label: LEFT, RIGHT, or TIE for equal values.", f"For the pair (left: {left}, right: {right}), answer only with the winning side LEFT or RIGHT; use TIE if neither is larger."]), expected, kind="comparison", left=left, right=right)
    elif category == "logical_relations":
        if variant == 0:
            forward = index % 2 == 0
            question = f"Is {name} before {third}?" if forward else f"Is {third} before {name}?"
            answer = f"Yes. {name} is before {other}, who is before {third}." if forward else f"No. {third} is after {name}."
            turn(choose([f"In a line, {name} is before {other}, and {other} is before {third}. {question}", f"Queue facts: {name} stands ahead of {other}; {other} stands ahead of {third}. {question}", f"Ordering rule: before is transitive, and nobody is before themselves. Facts: {name} before {other}; {other} before {third}. {question}", f"Three people are in a straight queue. {other} is behind {name}; {third} is behind {other}. {question}"]), answer, kind="order", edges=[[name, other], [other, third]], query=[name, third] if forward else [third, name], expected_boolean=forward)
        elif variant == 1:
            positive = known
            subject = name if positive else third
            conclusion = f"{subject} has a {color} badge."
            turn(choose([f"Every member of the {place} group has a {color} badge. {name} is a member. We have no membership information about {third}. Can we conclude that {conclusion[:-1]}?", f"Rule: all {place} group members have {color} badges. Known member: {name}. Nothing is stated about {third}. Does this establish that {conclusion[:-1]}?", f"The {place} group gives a {color} badge to each member. {name} belongs to it; {third}'s membership is unspecified. Is there enough information to say that {conclusion[:-1]}?", f"Assume every {place} group member has a {color} badge. Only {name} is confirmed as a member; {third} is not described. Is the conclusion '{conclusion}' supported?"]), f"Yes. {name} is a group member, so {name} has a {color} badge." if positive else f"We can't determine that; {third}'s membership and badge are not specified.", kind="membership_rule", members=[name], subject=subject, affirmative=positive)
        else:
            positive = known
            fact = f"The {place} light is on." if positive else f"The {place} door is open."
            answer = f"Yes. The light is on, so the rule says the door is open." if positive else "We can't tell. An open door does not establish that the light is on."
            question = "Does the rule establish that the door is open?" if positive else "Does this establish that the light is on?"
            turn(choose([f"Suppose the rule is: if the {place} light is on, its door is open. {fact} {question}", f"Given just this one-way condition, light on implies door open at the {place}. {fact} {question}", f"Rule for the {place}: whenever the light is on, the door is open. Fact: {fact} {question}", f"We know that an illuminated light at the {place} guarantees an open door. {fact} {question}"]), answer, kind="implication", fact="antecedent" if positive else "consequent", question="consequent" if positive else "antecedent", affirmative=positive)
    elif category == "arithmetic_add_subtract":
        a, b = rng.randint(10, 90), rng.randint(2, 25)
        c = rng.randint(1, min(a + b - 2, 35))
        result = a + b - c
        if variant == 0:
            user = choose([f"{name} has {a} {item}, receives {b} more, then gives away {c}. How many remain?", f"Stock note: start with {a} {item}; add {b}; remove {c}. What is the final count?", f"There were {a} {item} in the {place}. {name} brought {b} more and took away {c}. How many are there now?", f"Can you check the total? {a} {item} at first, another {b} delivered, and {c} sent out."])
            answer = f"{result} {item} remain."
        elif variant == 1:
            c = rng.randint(1, min(a, b))
            result = a + b - c
            user = choose([f"{name} scheduled two sessions of {a} and {b} minutes. A {c}-minute break is included in that total. How many minutes of activity remain after subtracting the break?", f"Two blocks last {a} minutes and {b} minutes in total. Subtract the included {c}-minute rest. How much active time is there?", f"Activity log for {name}: first block {a} minutes, second block {b} minutes; these times include a {c}-minute pause. What's the combined active time?", f"Combine blocks of {a} and {b} minutes, then remove {c} minutes of included downtime. How many active minutes does that leave?"])
            answer = f"There are {result} minutes of active time."
        else:
            user = choose([f"My balance starts at {a} vouchers. I receive {b} and spend {c}. What balance is left?", f"Work out the remaining vouchers: opening balance {a}, credit {b}, spending {c}.", f"I was holding {a} vouchers before {b} arrived. Then I used {c}. How many do I still have?", f"Voucher ledger: {a} carried in, {b} added, {c} redeemed. Find the closing balance."])
            answer = f"You have {result} vouchers left."
        turn(user, answer, kind="arithmetic_ledger", initial=a, received=b, removed=c, result=result)
    elif category == "arithmetic_groups":
        groups, size, extra = rng.randint(2, 12), rng.randint(2, 9), rng.randint(2, 16)
        if variant == 0:
            result = groups * size + extra
            user = choose([f"There are {groups} boxes with {size} {item} each, plus {extra} loose {item}. How many {item} altogether?", f"{name} packed {groups} equal boxes, each holding {size} {item}, and left {extra} unpacked. What is the total number of {item}?", f"Count the {item}: {groups} boxes of {size}, plus {extra} outside the boxes.", f"I have {groups} boxes containing {size} {item} apiece. Add {extra} more loose {item}. What's the combined count?"])
            answer = f"There are {result} {item} altogether."
            check = {"kind": "arithmetic_groups", "groups": groups, "size": size, "loose": extra, "remove": 0, "result": result}
        elif variant == 1:
            total = groups * size
            result = groups + extra
            user = choose([f"Split {total} {item} into groups of {size}. Then add {extra} already prepared groups. How many groups are there in total?", f"{name} makes groups of {size} from {total} {item}. There are also {extra} separate completed groups. What is the combined number of groups?", f"First divide {total} {item} into equal groups with {size} in each. Count those groups and {extra} existing groups together.", f"A batch of {total} {item} is packed {size} per group. Another {extra} groups are already packed. How many groups altogether?"])
            answer = f"There are {result} groups in total."
            check = {"kind": "arithmetic_division", "items": total, "group_size": size, "extra_groups": extra, "result": result}
        else:
            extra = rng.randint(2, groups * size - 2)
            result = groups * size - extra
            user = choose([f"We put {size} {item} on each of {groups} trays, then remove {extra} {item}. How many remain on the trays together?", f"Start with {groups} trays carrying {size} {item} each. Take away {extra} from the total. What is left?", f"The {place} has {groups} trays with {size} {item} per tray. After {extra} {item} are taken, what is the remaining count?", f"Multiply {groups} trays by {size} {item} on each, then account for {extra} removed {item}. How many are still there?"])
            answer = f"{result} {item} remain."
            check = {"kind": "arithmetic_groups", "groups": groups, "size": size, "loose": 0, "remove": extra, "result": result}
        turn(user, answer, **check)
    else:
        raise ValueError(category)

    messages = [{"role": "system", "content": system}, *turns]
    pair_id = f"{split}:{family}:{index // 2}" if category.endswith("information") else None
    return {
        "id": sha(dump(messages).encode()), "messages": messages,
        "source": VERSION, "retention": False, "split": split, "category": category,
        "family": family, "wording_id": f"{family}:{wording}",
        "pair_id": pair_id, "answerable": known if pair_id else None,
        "provenance": {"method": "independently_authored_procedural", "version": VERSION},
        "checks": checks,
    }


def check_record(row):
    messages = row["messages"]
    assert messages[0]["role"] == "system"
    assert [m["role"] for m in messages[1:]] == ["user", "assistant"] * ((len(messages) - 1) // 2)
    assert all(isinstance(m["content"], str) and m["content"].strip() for m in messages)
    assert all(len(m["content"]) < 1500 for m in messages)
    assert not any(marker in m["content"] for m in messages for marker in ("<|im_start|>", "<|im_end|>", "<|startoftext|>"))
    prompt = "\n".join(m["content"] for m in messages if m["role"] == "user")
    for check in row["checks"]:
        answer = messages[check["assistant_turn"] * 2]["content"]
        kind = check["kind"]
        if kind in ("grounded", "rewrite"):
            assert all(x in prompt and x in answer for x in check["anchors"])
        elif kind == "choice":
            assert check["selected"] in check["allowed"] and check["selected"] in answer and check["evidence"] in prompt
        elif kind == "repair":
            assert all(x in answer and x in prompt for x in check["required"])
            assert all(x not in answer for x in check["forbidden"])
        elif kind == "paired_grounding":
            assert check["evidence_present"] == check["answerable"]
            assert (check["relevant_fact"] in prompt) == check["answerable"]
            assert answer == check["expected_known"] if check["answerable"] else answer == check["expected_missing"]
        elif kind == "exact":
            assert answer == check["expected"]
        elif kind == "extract_rows":
            names = [r.split(": ")[0] for r in check["rows"] if r.split(": ")[1] == check["tag"]]
            assert answer == check["separator"].join(names)
        elif kind == "line_filter":
            assert answer.splitlines() == [line[len(check["prefix"]):] for line in check["lines"] if line.startswith(check["prefix"])]
        elif kind == "reverse":
            assert answer.split(check["separator"])[::-1] == check["values"]
        elif kind in ("rank", "clock_order"):
            values = answer.splitlines() if kind == "rank" else answer.split(" -> ")
            ranks = dict(check["entries"])
            assert set(values) == set(ranks) and len(values) == len(ranks)
            assert all(ranks[a] < ranks[b] for a, b in zip(values, values[1:]))
        elif kind == "copy":
            assert answer == check["payload"] and check["payload"] in prompt
        elif kind == "json":
            parsed = json.loads(answer)
            assert parsed == check["expected"] and set(parsed) == set(check["expected"])
            assert all(type(parsed[k]) is type(v) for k, v in check["expected"].items())
        elif kind == "mapping":
            assert answer == check["mapping"][check["selected"]] and len(answer) == 1
        elif kind == "membership":
            assert answer in ("yes", "no") and (answer == "yes") == (check["query"] in check["values"])
        elif kind == "comparison":
            assert answer in ("LEFT", "RIGHT", "TIE")
            assert {"LEFT": check["left"] > check["right"], "RIGHT": check["right"] > check["left"], "TIE": check["left"] == check["right"]}[answer]
        elif kind == "order":
            edges = {tuple(x) for x in check["edges"]}
            while True:
                closure = edges | {(a, d) for a, b in edges for c, d in edges if b == c}
                if closure == edges:
                    break
                edges = closure
            assert (tuple(check["query"]) in edges) == check["expected_boolean"]
            assert answer.startswith("Yes." if check["expected_boolean"] else "No.")
        elif kind == "membership_rule":
            established = check["subject"] in check["members"]
            assert check["affirmative"] == established
            assert answer.startswith("Yes." if established else "We can't determine")
        elif kind == "implication":
            established = check["fact"] == "antecedent" and check["question"] == "consequent"
            assert check["affirmative"] == established
            assert answer.startswith("Yes." if established else "We can't tell.")
        elif kind.startswith("arithmetic_"):
            if kind == "arithmetic_ledger":
                # Independent event-ledger reduction instead of the authoring expression.
                computed = sum([check["initial"], check["received"], -check["removed"]])
            elif kind == "arithmetic_groups":
                computed = sum([check["size"] for _ in range(check["groups"])]) + check["loose"] - check["remove"]
            else:
                quotient, remainder = divmod(check["items"], check["group_size"])
                assert remainder == 0
                computed = quotient + check["extra_groups"]
            assert computed == check["result"] and computed >= 0
            assert [int(n) for n in re.findall(r"\b\d+\b", answer)] == [computed]
        else:
            raise AssertionError(kind)
    return len(row["checks"])


def user_text(row):
    return normalize("\n".join(m["content"] for m in row["messages"] if m["role"] == "user"))


def grams(text):
    words = re.findall(r"\w+|[^\w\s]", text)
    return {tuple(words[i:i + 3]) for i in range(len(words) - 2)}


def validate_splits(rows):
    counts = collections.Counter()
    seen = set()
    for row in rows["train"] + rows["val"]:
        assert row["id"] not in seen, f"Duplicate dialogue: {row['id']}"
        seen.add(row["id"])
        counts["checked_records"] += 1
        counts["rule_checks"] += check_record(row)
    for key in ("family", "wording_id"):
        assert not {r[key] for r in rows["train"]} & {r[key] for r in rows["val"]}
    for key in POOL["train"]:
        assert not set(POOL["train"][key]) & set(POOL["val"][key])
    prompts = {s: {user_text(r) for r in data} for s, data in rows.items()}
    assert not prompts["train"] & prompts["val"], "Identical cross-split prompt"
    pairs = collections.defaultdict(list)
    for row in rows["train"] + rows["val"]:
        if row["pair_id"]:
            pairs[row["pair_id"]].append(row)
    for pair in pairs.values():
        assert len(pair) == 2 and {r["answerable"] for r in pair} == {False, True}
        a, b = [r["checks"][0] for r in pair]
        assert a["base"] == b["base"] and a["relevant_fact"] == b["relevant_fact"]
        counts["complete_known_unknown_pairs"] += 1
    # Conservative surface-overlap audit across ALL user-context prompts.
    # This detects high lexical overlap, not semantic or evaluation contamination.
    max_jaccard = 0.0
    nearest = None
    train_grams = [(r["id"], grams(user_text(r))) for r in rows["train"]]
    for row in rows["val"]:
        vg = grams(user_text(row))
        for tid, tg in train_grams:
            score = len(vg & tg) / max(1, len(vg | tg))
            if score > max_jaccard:
                max_jaccard, nearest = score, [tid, row["id"]]
    assert max_jaccard < 0.65, f"Near-duplicate cross-split prompts: {max_jaccard:.3f} {nearest}"
    return {**dict(counts), "cross_split_exact_prompt_overlap": 0,
            "cross_split_max_token_trigram_jaccard": round(max_jaccard, 6),
            "cross_split_max_overlap_pair_ids": nearest,
            "cross_split_near_duplicate_rejection_threshold": 0.65,
            "family_overlap": 0, "wording_id_overlap": 0, "entity_pool_overlap": 0}


def stats(data):
    assistants = [m["content"] for r in data for m in r["messages"] if m["role"] == "assistant"]
    return {
        "dialogues": len(data), "assistant_turns": len(assistants),
        "category_counts": dict(sorted(collections.Counter(r["category"] for r in data).items())),
        "family_counts": dict(sorted(collections.Counter(r["family"] for r in data).items())),
        "wording_shells": len({r["wording_id"] for r in data}),
        "known_rows_in_pairs": sum(r["answerable"] is True for r in data),
        "missing_rows_in_pairs": sum(r["answerable"] is False for r in data),
        "max_dialogue_characters": max(sum(len(m["content"]) for m in r["messages"]) for r in data),
        "mean_assistant_words": round(sum(len(a.split()) for a in assistants) / len(assistants), 2),
        "max_assistant_words": max(len(a.split()) for a in assistants),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "calibration/v1")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    permitted = [root / "tools/prepare_coherence_data.py", root / "tools/train_coherence.py"]
    system = system_from(permitted[0])
    rows = {"train": [], "val": []}
    seen_ids = set()
    duplicate_resamples = 0
    for split in rows:
        for category in CATEGORIES:
            variants = (0, 1) if split == "train" else (2,)
            per_family = 64 if split == "train" else 20
            for variant in variants:
                paired = category.endswith("information")
                width = 2 if paired else 1
                for group in range(per_family // width):
                    for attempt in range(100):
                        local_seed = int(sha(dump([args.seed, split, category, variant, group, attempt]).encode())[:16], 16)
                        batch = [authored_example(category, split, variant, group * width + offset, system, random.Random(local_seed)) for offset in range(width)]
                        if not seen_ids & {row["id"] for row in batch}:
                            rows[split].extend(batch)
                            seen_ids.update(row["id"] for row in batch)
                            break
                        duplicate_resamples += 1
                    else:
                        raise RuntimeError("Could not generate distinct examples for " + category)
        random.Random(args.seed + (0 if split == "train" else 1)).shuffle(rows[split])
    checks = validate_splits(rows)
    checks["deterministic_duplicate_resamples"] = duplicate_resamples
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to replace nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, data in rows.items():
        name = "train.jsonl" if split == "train" else "validation.jsonl"
        content = "".join(dump(row) + "\n" for row in data)
        (output / name).write_text(content)
        files[name] = {"sha256": sha(content.encode()), "bytes": len(content.encode()), "rows": len(data)}
    examples = ["# Independently authored calibration examples", "", "One deterministic example per scenario family. These are training and adaptation-validation data, not benchmark results.", ""]
    for split in ("train", "val"):
        for family in sorted({r["family"] for r in rows[split]}):
            family_rows = sorted([r for r in rows[split] if r["family"] == family], key=lambda r: r["id"])
            selected = [family_rows[0]]
            if family_rows[0]["pair_id"]:
                selected = [r for r in family_rows if r["pair_id"] == family_rows[0]["pair_id"]]
            for row in selected:
                examples.extend([f"## {split}: {family}", f"ID: `{row['id']}`", ""])
                for message in row["messages"][1:]:
                    examples.extend([f"**{message['role'].title()}**", "", "```text", message["content"], "```", ""])
    example_text = "\n".join(examples)
    (output / "EXAMPLES.md").write_text(example_text)
    files["EXAMPLES.md"] = {"sha256": sha(example_text.encode()), "bytes": len(example_text.encode())}
    manifest = {
        "version": VERSION, "seed": args.seed,
        "source_method": "Fresh human-readable procedural templates authored for this task; all scenario facts supplied in each dialogue; no externally sourced factual answers.",
        "inputs": [{"path": str(p.relative_to(root)), "sha256": sha(p.read_bytes()), "purpose": "SYSTEM/schema" if i == 0 else "training-interface provenance"} for i, p in enumerate(permitted)],
        "generator": {"path": "tools/prepare_calibration_data.py", "sha256": sha(Path(__file__).read_bytes())},
        "system": system, "files": files, "splits": {s: stats(data) for s, data in rows.items()},
        "partition": {"unit": "scenario family, entity pool, wording shell, and matched known/unknown pair", "pools": POOL, "families": FAMILIES, "validation_role": "Adaptation validation; never a blind capability benchmark."},
        "validation": checks,
        "not_consumed": ["evaluation/benchmark files", "model predictions", "acceptance tests", "public datasets", "tokenizer files", "network resources"],
        "limitations": ["Template-generated supplement; lexical and scenario breadth remains modest.", "No empirical model improvement is claimed and no training was run.", "Checks cover schema, explicit transformations, arithmetic, pairs, and lexical overlap; natural-language quality also needs human review.", "No tokenizer was read, so character and word lengths are reported instead of model token counts.", "Rows explicitly set retention:false; a trainer must honor that flag instead of relying exclusively on the legacy source-name rule.", "SYSTEM names Windows XP, but the examples do not establish hardware fit, runtime speed, or operating-system competence."],
    }
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    (output / "manifest.json").write_text(manifest_text)
    hashes = {**{name: meta["sha256"] for name, meta in files.items()}, "manifest.json": sha(manifest_text.encode())}
    (output / "SHA256SUMS").write_text("".join(f"{h}  {name}\n" for name, h in sorted(hashes.items())))
    print(json.dumps({"output": str(output), "splits": manifest["splits"], "validation": checks, "files": files}, indent=2))


if __name__ == "__main__":
    main()
