# Independently authored calibration examples

One deterministic example per scenario family. These are training and adaptation-validation data, not benchmark results.

## train: all_members_rule
ID: `0106433318195cb65762b87923df0e83298d64dba80a890b3e028e4dc9d95f85`

**User**

```text
Assume every Cedar nook group member has a red badge. Only Uma is confirmed as a member; Seth is not described. Is the conclusion 'Seth has a red badge.' supported?
```

**Assistant**

```text
We can't determine that; Seth's membership and badge are not specified.
```

## train: availability_slot
ID: `025a47ad0a569ca3384c404ea84a8b64a114d688e7d6c1a82f56e75f2f0122bd`

**User**

```text
I need a 15-minute slot for Greta. Available starts are 11:00, 12:00, and 13:00; Greta can attend only between 12:00 and 12:30. What fits?
```

**Assistant**

```text
Book 12:00; the 15-minute appointment fits the free window.
```

## train: boxes_plus_loose
ID: `07a0c37738c56a57a8848c429729afbc3fb790d887e59445b6ecaae024d3c7d7`

**User**

```text
There are 11 boxes with 3 cups each, plus 6 loose cups. How many cups altogether?
```

**Assistant**

```text
There are 39 cups altogether.
```

## train: cabinet_code_pair
ID: `01f888e9eb929deef112a9aca8ffd92357ca0b853099c57380228920010d82cd`

**User**

```text
Using this note, answer my question. The cabinet in the Moss porch contains trays.
What cabinet code does the note give?
```

**Assistant**

```text
No cabinet code is given in the note.
```

## train: cabinet_code_pair
ID: `664af7150d8773eac6b70baa7185efc5a936e32d63dc1e54a1b892de008dd977`

**User**

```text
Using this note, answer my question. The cabinet in the Moss porch contains trays. In this fictional exercise, the cabinet code is T1691.
What cabinet code does the note give?
```

**Assistant**

```text
The cabinet code is T1691.
```

## train: current_status_from_log
ID: `022e92e45831f1f2a17f2503637e5018cf32054c3c295f66c144aac3318960fc`

**User**

```text
Summarize where things stand now: cups delivered by Dev to the Cedar nook; later transferred by Greta to the Moss porch.
```

**Assistant**

```text
The cups are now at the Moss porch.
```

## train: destination_correction
ID: `077dff90953860acee2cab2620b34d9a0157903b812ca0bfd58a7224a49042c9`

**User**

```text
The Holly cabin is where Dev's books should go.
```

**Assistant**

```text
The destination is the Holly cabin.
```

**User**

```text
I changed my mind; use the Juniper office. What's the destination now?
```

**Assistant**

```text
They should go to the Juniper office.
```

## train: division_plus_groups
ID: `08145701a85713b396a2adb0fab9ee9ba1cdfd952249e507cc0fabdeb3972ad5`

**User**

```text
Tess makes groups of 8 from 80 mugs. There are also 10 separate completed groups. What is the combined number of groups?
```

**Assistant**

```text
There are 20 groups in total.
```

## train: explicit_list_membership
ID: `11ba9c7f32aa3da98f5bdccc516bd4c8307fe2d9de131e8d94b4154d2aa72122`

**User**

```text
Check membership: scarves; allowed entries: scarves, socks. Output just yes if present, otherwise no.
```

**Assistant**

```text
yes
```

## train: item_count_object
ID: `01246a197748a436c61767e1b5f0c246ac7d349788005440dfb09642f1fee2aa`

**User**

```text
Produce only valid JSON for books, quantity 57. Keys must be item and count; use a string and an integer respectively.
```

**Assistant**

```text
{"count": 57, "item": "books"}
```

## train: object_clarification
ID: `015f15c6fee9564b66a0049c8239148a21520937b08ef658111ede9d367a2670`

**User**

```text
I put the books in the Willow room and the keys in the Cedar nook. Remind me where the books are.
```

**Assistant**

```text
The books are at the Willow room.
```

**User**

```text
Let me correct that: I need the keys.
```

**Assistant**

```text
The keys are at the Cedar nook.
```

## train: parcel_owner_pair
ID: `03c89915f312e99df5d99eaa1297f533d62e4be2b897a22192e5ee7a9cc22843`

**User**

```text
Using this note, answer my question. A parcel containing towels is at the Laurel shed. The parcel belongs to Ada.
Who owns the parcel?
```

**Assistant**

```text
The parcel belongs to Ada.
```

## train: parcel_owner_pair
ID: `a42b28a3a32f405f0e21f82fb3739786617879c69190a482337ea370f584d184`

**User**

```text
Using this note, answer my question. A parcel containing towels is at the Laurel shed.
Who owns the parcel?
```

**Assistant**

```text
The note doesn't identify the parcel's owner.
```

## train: person_room_object
ID: `034eab9e4ae411d493d37704ccbc56c61747cb578b0b7209a308df5c984dec03`

**User**

```text
Output exactly two JSON string fields, name and room, using this note: Flynn is assigned to the Acorn hall. No prose.
```

**Assistant**

```text
{"name": "Flynn", "room": "Acorn hall"}
```

## train: pickup_rule
ID: `0468bdabf4390fa9213177073a1d6ff159bc1c8e914a0f9b60476e28356c0c89`

**User**

```text
The cups are available at the Thistle booth with a pass, or the Cedar nook without one. I have no pass; help me choose the right point.
```

**Assistant**

```text
Go to the Cedar nook.
```

## train: plural_referent
ID: `04e3902fe98d75460b46dd3df44d66081f5757379ce4acefb7f98d3708c80568`

**User**

```text
I've picked up the towels. Their assigned spot is the Laurel shed.
```

**Assistant**

```text
Understood.
```

**User**

```text
Where do these go again?
```

**Assistant**

```text
Put the towels in the Laurel shed.
```

## train: polite_request_rewrite
ID: `018163767e9538a3526399945ed7749ed6a9e467b932fe5ed9289858fe8b408b`

**User**

```text
Could you soften the wording? "Put the keys beside the towels."
```

**Assistant**

```text
Could you please put the keys beside the towels?
```

## train: punctuation_payload
ID: `025ef701e1f61ca178804913e9d6eb71b467328c703c34f693e1c12b2bdc9975`

**User**

```text
Copy the text after TEXT exactly, with no prefix or quotation marks.
TEXT
TU222: crayons!
```

**Assistant**

```text
TU222: crayons!
```

## train: quoted_spacing_payload
ID: `026401a697368b8af9cb7e50c9e420ea46ed3fc9eb3fa1b3ce78dfad4727897e`

**User**

```text
Give only the contents of these quotes. Don't change capitalization or spacing: "Nell  -  TU820"
```

**Assistant**

```text
Nell  -  TU820
```

## train: rank_order
ID: `00ed0fb656a3c00bb96d81e182b637369fbde6b7a03758bb9a54e841b121b211`

**User**

```text
Sort by rank from lowest to highest. Return names only, one per line:
Ines: rank 23
Otis: rank 3
Seth: rank 20
```

**Assistant**

```text
Otis
Seth
Ines
```

## train: reverse_sequence
ID: `04224bc4b9dc0a1063b52323ad5c956fbbcb8556d8d70aee027c3aee904ff969`

**User**

```text
Reverse this sequence. Output only the reversed values joined by ' > ': mugs, keys, books, towels
```

**Assistant**

```text
towels > books > keys > mugs
```

## train: rows_matching_label
ID: `01028fce950cca108467574bc143a4c0afab4dc77e34b243f5fca18bd72eeb1d`

**User**

```text
Select the collect names from this list. Keep the original order and output comma-separated names only:
Ada: wait
Wade: collect
Vera: collect
```

**Assistant**

```text
Wade, Vera
```

## train: same_location_relation
ID: `00e91ada0db5cf1630f762a8e550e4b7a1dd422b40b9be549b41ba1d121da08d`

**User**

```text
Zeke leaves the maps in the Holly cabin, and Rosa leaves the keys there too.
```

**Assistant**

```text
Got it.
```

**User**

```text
Which place holds the keys?
```

**Assistant**

```text
Rosa's keys are in the Holly cabin.
```

## train: selected_note_fields
ID: `02c5da7c096d6c7012bf59b3a54e25e95c40a47de3ab4caa65d3bbfb28b641f9`

**User**

```text
Fields: contents socks; room Clover loft; owner Ada. Output just 'owner | room', replacing those labels with their values.
```

**Assistant**

```text
Ada | Clover loft
```

## train: session_minutes
ID: `0ad2a0c58813ecd5d5938803a9e51d0ff5dcea2ada29511cc844e87007d512c3`

**User**

```text
Combine blocks of 53 and 20 minutes, then remove 10 minutes of included downtime. How many active minutes does that leave?
```

**Assistant**

```text
There are 63 minutes of active time.
```

## train: snack_preference_pair
ID: `00229cdfc72477e9e4ff3c0e58409263181e76ae9aefda12aef33f8cdb8ab827`

**User**

```text
I have this information: Ada keeps the towels in the Acorn hall. Ada's favorite snack is crackers.
Based on it, what is Ada's favorite snack?
```

**Assistant**

```text
Ada's favorite snack is crackers.
```

## train: snack_preference_pair
ID: `5f2274bc53e122436cfb5c0ce68ac8a593c7e694db69a5db353e532cd1e73cea`

**User**

```text
I have this information: Ada keeps the towels in the Acorn hall.
Based on it, what is Ada's favorite snack?
```

**Assistant**

```text
The note doesn't say what Ada's favorite snack is.
```

## train: stock_movements
ID: `016c04a58f98083b588298357908566283335154ae8a1bf93a889a5334790d7e`

**User**

```text
Can you check the total? 11 combs at first, another 18 delivered, and 1 sent out.
```

**Assistant**

```text
28 combs remain.
```

## train: story_book_title_pair
ID: `03b029a3c992f7e396c9f5207675f090f6b8c2a2db43ad84ea6b5372008303bb`

**User**

```text
I have this information: In an invented story, Uma reads a book at the Holly cabin. The book is called The Yellow Keys.
Based on it, what is the title of the book in this story?
```

**Assistant**

```text
The book is called The Yellow Keys.
```

## train: story_book_title_pair
ID: `58d953330f719fb5ba99abf8b2a4dd80ec7f2bae4ee9a61f7578b062ce18a446`

**User**

```text
I have this information: In an invented story, Uma reads a book at the Holly cabin.
Based on it, what is the title of the book in this story?
```

**Assistant**

```text
The excerpt doesn't give the book's title.
```

## train: story_captain_pair
ID: `0c3036c722ab6ccf1e42f6b59f4dcabf5b991c27aff3f249ad8d8245e4653bcd`

**User**

```text
What can you tell me from these details? In this invented story, a ship arrives at the port near the Juniper office with scarves. Its captain is Flynn.
Who captains the ship in this story?
```

**Assistant**

```text
Flynn is the ship's captain.
```

## train: story_captain_pair
ID: `324fdf7b5e604f14efb5ccd2d147ab29df55e3862da42f26092a83dd6cf03761`

**User**

```text
What can you tell me from these details? In this invented story, a ship arrives at the port near the Juniper office with scarves.
Who captains the ship in this story?
```

**Assistant**

```text
The story excerpt doesn't identify the ship's captain.
```

## train: supplied_code_mapping
ID: `0280b4d994009d6cb29d0aa4f34729dae77b6ce8197793f0917641cff44670c0`

**User**

```text
Use this mapping: Seth->A; Milo->B; Hugo->C. Return only Milo's code.
```

**Assistant**

```text
B
```

## train: transitive_order
ID: `057f3435bab61daa65d92490348d86008fc225280e1ab17f0d17486005dc344a`

**User**

```text
Ordering rule: before is transitive, and nobody is before themselves. Facts: Zeke before Rosa; Rosa before Vera. Is Zeke before Vera?
```

**Assistant**

```text
Yes. Zeke is before Rosa, who is before Vera.
```

## train: two_plan_facts
ID: `038c4ca360aa317f9437bf4832385bd1a8eb6659695d730acf36bcacc0ed720f`

**User**

```text
Summarize this in one sentence: Xena will bring the combs. The meeting will be in the Daisy annex.
```

**Assistant**

```text
Xena will bring the combs to the meeting in the Daisy annex.
```

## train: verbose_note_simplify
ID: `03dd5231d927e04e611b8bb64a0f8c4776f534913d457143830ed6ee4e947f1d`

**User**

```text
Can you say this more simply? "The location at which Vera left the cups is the Laurel shed."
```

**Assistant**

```text
Vera left the cups in the Laurel shed.
```

## train: weekend_activity_pair
ID: `07c06150166e117828587ea2646ebb87864d4e2d5155788fa4ed2545669f3ea1`

**User**

```text
What can you tell me from these details? Lena brought lamps to the Willow room.
What does Lena plan to do on Saturday?
```

**Assistant**

```text
The note doesn't give Lena's Saturday plan.
```

## train: weekend_activity_pair
ID: `643ed87d5ee4b2e09f0527377ffd56fb950f82d0d97e526c562fe6a460ced0df`

**User**

```text
What can you tell me from these details? Lena brought lamps to the Willow room. Lena plans to spend Saturday gardening.
What does Lena plan to do on Saturday?
```

**Assistant**

```text
Lena plans to spend Saturday gardening.
```

## val: clock_order
ID: `0cdd3583e618ead222c1964853b2b2b0061528c3f452fa40daccc9d61d2b34d6`

**User**

```text
Use these 24-hour appointments to give the chronological route. Format the place names as 'first -> second -> third' without a preface:
Tern hut: 12:45
Ibis den: 10:25
Dove chamber: 09:20
```

**Assistant**

```text
Dove chamber -> Ibis den -> Tern hut
```

## val: elliptical_other_person
ID: `1cbd8652693631812340ec360e7eb159aa0c2894c1f05d9c3ce88a8f2e05b1ae`

**User**

```text
There are two selections: umber for Keiko, indigo for Lucian. Start with Keiko's color.
```

**Assistant**

```text
Keiko chose umber.
```

**User**

```text
How about the choice for Lucian?
```

**Assistant**

```text
Lucian chose indigo.
```

## val: event_and_stated_reason
ID: `031a78f9a3a11f5ad77314e8a5dc2c8aaf65efe983bc877ff298b5bd17a20c4d`

**User**

```text
Make a brief recap containing the change and its cause: the first room was locked, so Keiko's group moved to the Wren pavilion.
```

**Assistant**

```text
Keiko's group moved to the Wren pavilion because the first room was locked.
```

## val: keep_line_filter
ID: `043d5d31d72f05cfbabc66068be0e66f8c394a05108abf1229b4bb55a204150f`

**User**

```text
Remove skip lines and strip 'keep: ' from the others. Return the resulting lines alone:
skip: erasers
skip: mirrors
keep: gloves
keep: coins
```

**Assistant**

```text
gloves
coins
```

## val: larger_side_label
ID: `036e5392fb82999cc754f8ba0ebdf0a876dad22b97869a06fec4e2aef75130da`

**User**

```text
Choose the larger side for left 42 and right 34. Give a single label: LEFT, RIGHT, or TIE for equal values.
```

**Assistant**

```text
LEFT
```

## val: marked_two_line_payload
ID: `0912783a131525ae6c1765146a41a34564ea56a8c006855ad7cf77d6458d1b12`

**User**

```text
Return only the two marked lines, with their original case, symbols, and line break:
BEGIN
BOWLS
Bruno / VE537
END
```

**Assistant**

```text
BOWLS
Bruno / VE537
```

## val: message_draft_from_intent
ID: `056be09dfe90cd58f4e99e6c13526e0b1f5925d4ca14c10291e4d577abb9a9b6`

**User**

```text
I want to ask Alma to leave the brushes at the Swan passage. What short message could I send?
```

**Assistant**

```text
Hi Alma, could you leave the brushes at the Swan passage?
```

## val: milestone_date_pair
ID: `0edbd23e9032348441a162ef0e012d0764f82f832861f14e9e7fadd998984c37`

**User**

```text
Read this short brief: Keiko is coordinating the Kestrel lodge project. The milestone is scheduled for October 13.
What date is the milestone scheduled for? Use the brief as your source.
```

**Assistant**

```text
The milestone is scheduled for October 13.
```

## val: milestone_date_pair
ID: `be9a78d9ff23889580c4aa34788eaadd50c8a09889b26685d0f6ef348e0c9c2a`

**User**

```text
Read this short brief: Keiko is coordinating the Kestrel lodge project.
What date is the milestone scheduled for? Use the brief as your source.
```

**Assistant**

```text
The milestone date cannot be determined from this brief.
```

## val: music_choice_pair
ID: `111b8ca90c63b5acba65c6743f704717295c64dec891913686b3132f2a9f326a`

**User**

```text
Work from this information alone: Keiko signed up in the Lark corner.
My question is: Which music did Keiko choose for this event?
```

**Assistant**

```text
Keiko's music choice isn't included in this note.
```

## val: music_choice_pair
ID: `284920d1fb19833443e503a8e8990caaa68629988f56bf3b31eaf669bfea675a`

**User**

```text
Work from this information alone: Keiko signed up in the Lark corner. For this event, Keiko chose jazz.
My question is: Which music did Keiko choose for this event?
```

**Assistant**

```text
Keiko chose jazz for this event.
```

## val: preference_replacement
ID: `5095d59100f862fcd798d0bcca22581d30f3e5417f62f67d9e8acfea5d7a2d33`

**User**

```text
For this order, Gus selected violet.
```

**Assistant**

```text
Gus's selected color is violet.
```

**User**

```text
Update from Gus: teal instead. Which color is current?
```

**Assistant**

```text
Gus's current choice is teal.
```

## val: room_constraints
ID: `041f288173bceafc7f0fb935e0f984517359ac1fa5b71ee77d87aa91387b6c80`

**User**

```text
Our group has 5 members and needs a quiet space. Options: Swift gallery (4 seats, quiet); Robin bay (5 seats, quiet); Finch terrace (8 seats, noisy). Where can we meet?
```

**Assistant**

```text
Use the Robin bay; it is quiet and has enough seats.
```

## val: story_courier_destination_pair
ID: `02ad792ea94631240b7b854f422a8af09c05c18b1431dc11c9e506934a1d84b6`

**User**

```text
Read this short brief: In a made-up tale, Isla is a courier carrying gloves. The courier's destination is the Lark corner.
Where is the courier headed in the tale? Use the brief as your source.
```

**Assistant**

```text
The courier is headed to the Lark corner.
```

## val: story_courier_destination_pair
ID: `0c86affd37f571ef176e543d3cddc40eb696e834d4e77449a092cc8ae9e0eed9`

**User**

```text
Read this short brief: In a made-up tale, Isla is a courier carrying gloves.
Where is the courier headed in the tale? Use the brief as your source.
```

**Assistant**

```text
The supplied passage leaves the courier's destination unstated.
```

## val: sufficient_condition
ID: `00071578095f3302d2d1663c40cfbc91dafdb95a0833cc891d1b8c2d00a6fddf`

**User**

```text
Suppose the rule is: if the Swan passage light is on, its door is open. The Swan passage light is on. Does the rule establish that the door is open?
```

**Assistant**

```text
Yes. The light is on, so the rule says the door is open.
```

## val: task_boolean_object
ID: `30078877e05c6a2d3f018d3c179024b428d9684fc1927734d9ed04727192d349`

**User**

```text
Serialize this entry with precisely two keys, task and ready: 'move bowls' is ready. Use a JSON boolean for ready and no surrounding explanation.
```

**Assistant**

```text
{"ready": true, "task": "move bowls"}
```

## val: trays_minus_removed
ID: `00ab157b16ce729a0fbe5c41962ef6402c23101fdd1364eb85632c9529968b75`

**User**

```text
The Tern hut has 11 trays with 9 mirrors per tray. After 96 mirrors are taken, what is the remaining count?
```

**Assistant**

```text
3 mirrors remain.
```

## val: voucher_balance
ID: `0d836434967ada9ba0d0017c5b812b609a4337eaab3ce4e4b2f747bad5a8c85b`

**User**

```text
My balance starts at 87 vouchers. I receive 18 and spend 29. What balance is left?
```

**Assistant**

```text
You have 76 vouchers left.
```
