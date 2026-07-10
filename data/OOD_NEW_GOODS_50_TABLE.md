# OOD New-Goods Evaluation Suite (50 Goods)

> Evaluation-only: do not use this suite for training, reward construction, hyperparameter tuning, or checkpoint selection.

## Design summary

| Field | Value |
|---|---:|
| New categories | 10 |
| New goods | 50 |
| New attribute names | 100 |
| Unordered goods pairs | 1,225 |
| Configurations per pair | 8 |
| Structural cases | 9,800 |
| X/Y prompts | 19,600 |
| Deterministic generation seed | 20260710 |

All category names, good names, attribute names, and complete attribute-value strings were checked for exact overlap against `everyday_goods_full.json`; all overlap counts are zero.

## Goods and attributes

| # | Category | Good | Attribute 1 (low → high) | Attribute 2 (low → high) |
|---:|---|---|---|---|
| 1 | Outdoor & Camping Equipment | Camping lantern | **Illumination reach:** tent-interior reach → full-campsite reach → wide-area trail reach | **Weather sealing grade:** dry-condition housing → rain-resistant housing → storm-sealed housing |
| 2 | Outdoor & Camping Equipment | Hiking backpack | **Trail load capacity:** light day-hike load → full-day trail load → multi-day expedition load | **Back support system:** unpadded back panel → contoured foam support → ventilated suspension support |
| 3 | Outdoor & Camping Equipment | Sleeping bag | **Overnight temperature rating:** mild-night rating → cool-night rating → subzero-night rating | **Insulation fill grade:** basic synthetic fill → lofted synthetic fill → premium down-alternative fill |
| 4 | Outdoor & Camping Equipment | Portable camping stove | **Burner output range:** low-output simmer burner → general-purpose camp burner → high-output rapid-boil burner | **Wind protection system:** exposed flame ring → partial wind collar → integrated windscreen chamber |
| 5 | Outdoor & Camping Equipment | Field binoculars | **Distant image clarity:** basic central clarity → edge-corrected clarity → high-definition full-field clarity | **Viewing stability aid:** freehand-only viewing → textured steady-grip body → image-stabilized viewing |
| 6 | Home Improvement & Hardware | Cordless drill | **Fastening torque capability:** light assembly torque → general repair torque → heavy fastening torque | **Work-session battery endurance:** short task endurance → half-day project endurance → full-day project endurance |
| 7 | Home Improvement & Hardware | Claw hammer | **Strike balance quality:** front-heavy balance → neutral workshop balance → precision-tuned balance | **Impact vibration damping:** rigid impact transfer → rubberized impact reduction → multi-layer vibration isolation |
| 8 | Home Improvement & Hardware | Adjustable wrench | **Jaw adjustment precision:** coarse jaw adjustment → fine-thread jaw adjustment → zero-play precision adjustment | **Metal corrosion protection:** unfinished steel surface → protective chrome coating → marine-grade protective coating |
| 9 | Home Improvement & Hardware | Laser distance meter | **Measurement distance span:** single-room distance span → whole-house distance span → large-site distance span | **Distance reading tolerance:** centimeter-level tolerance → five-millimeter tolerance → millimeter-level tolerance |
| 10 | Home Improvement & Hardware | Folding step ladder | **Supported working load:** light household load → standard adult work load → heavy equipment work load | **Footing stabilization design:** plain rubber feet → wide anti-slip feet → self-leveling stabilizer feet |
| 11 | Kitchen Appliances & Tools | Countertop air fryer | **Cooking basket volume:** single-serving basket → small-family basket → large-family basket | **Hot-air circulation control:** fixed fan circulation → adaptive fan circulation → dual-zone precision circulation |
| 12 | Kitchen Appliances & Tools | Electric kettle | **Water heating speed:** standard heating cycle → rapid heating cycle → ultrafast heating cycle | **Brew temperature selection:** boil-only selection → three preset temperatures → degree-by-degree temperature control |
| 13 | Kitchen Appliances & Tools | Hand blender | **Blending motor strength:** soft-food motor → general blending motor → heavy-duty crushing motor | **Food-prep attachment range:** blending shaft only → shaft plus whisk → multi-tool preparation set |
| 14 | Kitchen Appliances & Tools | Digital kitchen scale | **Ingredient weight resolution:** whole-gram resolution → half-gram resolution → tenth-gram resolution | **Maximum platform load:** small-bowl load → mixing-bowl load → bulk-ingredient load |
| 15 | Kitchen Appliances & Tools | Chef's knife | **Blade edge retention:** frequent-sharpening edge → extended-use edge → professional long-life edge | **Cutting grip ergonomics:** straight basic grip → contoured balanced grip → custom-fit anti-fatigue grip |
| 16 | Fitness & Sports Gear | Yoga exercise mat | **Joint cushioning density:** thin firm cushioning → medium balanced cushioning → thick impact-absorbing cushioning | **Pose traction performance:** basic dry traction → sweat-resistant traction → professional non-slip traction |
| 17 | Fitness & Sports Gear | Resistance band set | **Exercise tension range:** light tension range → light-to-heavy tension range → rehab-to-athletic tension range | **Band snap resistance:** standard latex resistance → reinforced latex resistance → sleeved break-resistant construction |
| 18 | Fitness & Sports Gear | Cast-iron kettlebell | **Weight casting accuracy:** approximate cast weight → calibrated training weight → competition-certified weight | **Lift grip comfort:** rough narrow handle → smoothed wide handle → ergonomic competition handle |
| 19 | Fitness & Sports Gear | Speed jump rope | **Handle rotation smoothness:** friction rotation → bushed rotation → sealed-bearing rotation | **Rope length adjustment:** cut-to-length adjustment → tool-adjusted length → instant tool-free adjustment |
| 20 | Fitness & Sports Gear | Muscle foam roller | **Massage firmness selection:** single soft firmness → single medium firmness → variable-zone firmness | **Surface pressure contour:** smooth surface → shallow massage ridges → multi-depth trigger contours |
| 21 | Pet Care & Animal Supplies | Dry cat food bag | **Animal protein proportion:** basic protein proportion → high protein proportion → meat-first protein proportion | **Ingredient source disclosure:** general ingredient listing → named ingredient sources → fully traceable ingredient sources |
| 22 | Pet Care & Animal Supplies | Dog walking leash | **Sustained pull tolerance:** small-dog pull tolerance → medium-dog pull tolerance → large-dog pull tolerance | **Collar clasp security:** basic spring clasp → locking swivel clasp → double-secured climbing clasp |
| 23 | Pet Care & Animal Supplies | Pet travel carrier | **Airflow panel coverage:** single ventilation panel → three-sided ventilation → full-surround ventilation | **Carrier frame rigidity:** soft unframed body → reinforced flexible frame → impact-resistant rigid frame |
| 24 | Pet Care & Animal Supplies | Aquarium water filter | **Tank flow adjustment:** fixed water flow → three-step water flow → continuous precision flow | **Water filtration stages:** single mechanical stage → mechanical plus carbon stages → three-stage biological filtration |
| 25 | Pet Care & Animal Supplies | Cat scratching tower | **Scratching column height:** kitten-height column → adult-cat stretch column → full-height climbing column | **Tower base stability:** compact lightweight base → wide weighted base → anti-tip anchored base |
| 26 | Gardening & Plant Care | Garden hand trowel | **Digging blade rigidity:** flexible stamped blade → reinforced steel blade → forged rigid blade | **Soil release treatment:** untreated blade surface → polished release surface → non-stick soil-shedding coating |
| 27 | Gardening & Plant Care | Pruning shears | **Branch cutting diameter:** thin-stem cutting → medium-branch cutting → thick-branch cutting | **Handle return mechanism:** manual handle reopening → basic coil-spring return → adjustable assisted return |
| 28 | Gardening & Plant Care | Garden watering can | **Pour stream control:** open-spout stream → removable shower rose → precision multi-pattern rose | **Water reservoir volume:** balcony-plant volume → patio-garden volume → large-bed garden volume |
| 29 | Gardening & Plant Care | Seed propagation tray | **Reusable cell durability:** single-season cells → multi-season reinforced cells → rigid nursery-grade cells | **Root drainage layout:** single drain opening → multi-hole drainage → air-pruning drainage channels |
| 30 | Gardening & Plant Care | Expandable garden hose | **Hose kink resistance:** basic flexible hose → braided kink-resistant hose → self-straightening anti-kink hose | **Water pressure tolerance:** low-pressure watering → household mains pressure → high-pressure outdoor use |
| 31 | Travel & Mobility Accessories | Hard-shell travel suitcase | **Shell impact resistance:** light-impact shell → reinforced travel shell → high-impact composite shell | **Wheel movement quality:** two fixed wheels → four spinner wheels → silent precision-bearing spinners |
| 32 | Travel & Mobility Accessories | Travel neck cushion | **Neck support structure:** soft wrap support → contoured side support → adjustable orthopedic support | **Packed travel size:** full-size carry → compressible pouch size → ultracompact roll size |
| 33 | Travel & Mobility Accessories | Packing organizer cube set | **Clothing compression ability:** organization only → zippered light compression → double-zip maximum compression | **Cube seam durability:** single-stitched seams → reinforced double seams → taped abrasion-resistant seams |
| 34 | Travel & Mobility Accessories | Portable luggage scale | **Bag weighing capacity:** cabin-bag capacity → checked-bag capacity → oversize-baggage capacity | **Displayed weight accuracy:** half-kilogram accuracy → hundred-gram accuracy → fifty-gram accuracy |
| 35 | Travel & Mobility Accessories | Passport document organizer | **Travel document capacity:** single-passport capacity → couple travel capacity → family document capacity | **Personal data protection:** standard fabric lining → RFID-blocking lining → shielded locking enclosure |
| 36 | Consumer Electronics | Portable Bluetooth speaker | **Room audio projection:** personal desk projection → full-room projection → outdoor gathering projection | **Continuous playback duration:** short-session playback → all-day playback → multi-day playback |
| 37 | Consumer Electronics | Wireless computer mouse | **Cursor tracking precision:** basic office tracking → high-resolution tracking → professional adjustable tracking | **Click switch durability:** standard click lifespan → extended click lifespan → rated esports click lifespan |
| 38 | Consumer Electronics | Portable power bank | **Stored charging energy:** single-phone recharge → multiple-phone recharges → laptop-capable energy reserve | **Simultaneous charging outlets:** one charging outlet → two charging outlets → multi-device fast-charge hub |
| 39 | Consumer Electronics | USB conference webcam | **Video image definition:** standard-definition video → full-HD video → ultra-HD video | **Dim-room image handling:** basic automatic exposure → low-light enhancement → sensor-level night correction |
| 40 | Consumer Electronics | Digital e-book reader | **Reading display sharpness:** entry-level text sharpness → print-like text sharpness → high-density premium sharpness | **Offline library capacity:** small personal library → large personal library → archive-scale library |
| 41 | Arts & Creative Materials | Watercolor paint palette | **Paint pigment concentration:** student pigment concentration → artist pigment concentration → professional pigment concentration | **Color blending consistency:** variable blending behavior → reliable blending behavior → studio-grade uniform blending |
| 42 | Arts & Creative Materials | Mixed-media sketchbook | **Drawing paper weight:** light sketch paper → medium mixed-media paper → heavy wet-media paper | **Paper surface tooth:** smooth pencil surface → medium all-purpose tooth → deep charcoal-friendly tooth |
| 43 | Arts & Creative Materials | Acrylic artist brush set | **Bristle shape recovery:** basic synthetic recovery → resilient artist recovery → precision shape-memory recovery | **Brush profile variety:** essential three profiles → expanded studio profiles → complete specialty profile range |
| 44 | Arts & Creative Materials | Reusable modeling clay kit | **Clay color assortment:** six-color assortment → twelve-color assortment → full-spectrum color assortment | **Sculpted shape retention:** soft temporary retention → firm project retention → detail-preserving long retention |
| 45 | Arts & Creative Materials | Embroidery beginner kit | **Included thread variety:** basic color bundle → expanded color bundle → shaded full-palette bundle | **Stitching hoop stability:** basic plastic hoop → locking wooden hoop → tension-controlled hoop frame |
| 46 | Learning & Educational Tools | Desktop world globe | **Printed map detail:** countries-only map detail → cities-and-terrain detail → reference-atlas map detail | **Globe rotation mechanism:** basic spindle rotation → smooth axis rotation → dual-axis precision rotation |
| 47 | Learning & Educational Tools | Home science experiment kit | **Experiment topic breadth:** single-topic experiments → multi-topic laboratory set → cross-discipline project library | **Procedure explanation clarity:** brief instruction cards → illustrated step guidance → concept-rich guided curriculum |
| 48 | Learning & Educational Tools | Magnetic construction tile set | **Tile connection strength:** light magnetic connection → reinforced magnetic connection → load-bearing magnetic connection | **Construction shape assortment:** basic square-triangle set → expanded geometric set → advanced architectural shape set |
| 49 | Learning & Educational Tools | Language vocabulary card set | **Vocabulary topic coverage:** survival vocabulary → daily conversation vocabulary → comprehensive thematic vocabulary | **Review sequencing system:** unordered card stack → level-grouped review → spaced-repetition review indexing |
| 50 | Learning & Educational Tools | Beginner optical microscope | **Specimen enlargement range:** low classroom enlargement → standard biology enlargement → high-detail cellular enlargement | **Image focus adjustment:** single coarse focus → coarse and fine focus → precision dual-speed focus |
