# Data

Lumina is trained only on CC0 panoramas. Nothing research-only (Laval, SUN360) and no ImageNet
weights are used. Every HDRI below is listed by id with its source; the per-asset URL, page,
authors and md5 are in `data/hdri/<source>/manifest.json` after a download.

Regenerate this file: `uv run python -m lumina_train.datadoc` (after
`uv run python -m lumina_train.download polyhaven` and `... download ambientcg --count 48`, then
`uv run python -m lumina_train.split`).

## Sources

### Poly Haven (training, validation, test)

- API: `https://api.polyhaven.com/assets?t=hdris`
- Licence: CC0 1.0 (https://polyhaven.com/license)
- Downloaded: 2026-11-13 (UTC), resolution 1k, format .hdr, User-Agent `lumina-hdri-fetch/0.1`
- The API listed 997 HDRIs; 997 were downloaded (278 indoor, 719 outdoor).

- Licence page: https://polyhaven.com/license (CC0: "You can use our assets for any purpose").
  The API terms (https://github.com/Poly-Haven/Public-API/blob/master/ToS.md) ask for a unique
  User-Agent naming the software; the downloader sends `lumina-hdri-fetch/0.1` and nothing personal,
  runs at most 4 downloads at once, retries with backoff and skips files already present.
- Asset pages: `https://polyhaven.com/a/<id>`.

### ambientCG (out-of-distribution test only)

- API: `https://ambientcg.com/api/v2/full_json?type=HDRI&limit=1000&include=downloadData`
- Licence: CC0 1.0 (https://docs.ambientcg.com/license/)
- Downloaded: 2026-11-13 (UTC), resolution 1k, format .exr, User-Agent `lumina-hdri-fetch/0.1`
- The API listed 439 HDRIs; 48 were downloaded (23 indoor, 25 outdoor).

- Licence page: https://docs.ambientcg.com/license/ ("All ambientCG assets are provided under the
  Creative Commons CC0 1.0 Universal License").
- Selection: all of Indoor Environment HDRI, then seed 20261113 sample of the others. The 1K zip is downloaded and only its `_HDR.exr` is kept.
- Asset pages: `https://ambientcg.com/a/<id>`.

## Split

`training/splits/split.json` (seed 20261113). Split by **scene**, never by crop: HDRIs
sharing GPS coordinates to 0.01 degrees or an id stem (time-of-day words and trailing numbers
removed) form one group (551 groups), and whole groups are assigned so indoor and
outdoor are each split about 80/10/10. "indoor"/"outdoor" is Poly Haven's `environment` attribute
(ambientCG: the "Indoor Environment HDRI" category).

| Split | Total | Indoor | Outdoor |
|---|---|---|---|
| train | 792 | 219 | 573 |
| val | 102 | 28 | 74 |
| test | 103 | 31 | 72 |
| ood | 48 | 23 | 25 |

## Synthetic camera model

Each training sample is made on the fly (`lumina_train.params`, `camera`, `isp`, `synth`):

| Step | Distribution |
|---|---|
| Yaw | uniform over 360 degrees |
| Pitch | normal, mean 0, sd 15 degrees, clipped to +/-40 |
| Roll | normal, mean 0, sd 3 degrees, clipped to +/-10 |
| Vertical FOV | uniform 40 to 75 degrees; 128 x 96 (4:3), rendered at 2x, box-filtered |
| Mirror | 50 %: image flipped left-right and label `mirror_x`'d (a mirror-image world) |
| Lens blur | Gaussian, sigma uniform 0 to 1.2 output pixels |
| Vignetting | 1 - v r^2, v uniform 0 to 0.35, r = 1 at the corners |
| Auto-exposure | p-th luminance percentile to 0.25 (p in 0.5..0.85), then 2^ev, ev in -1..1 |
| White balance | blackbody T seen by a 6500 K-balanced camera; T 3000-9000 K (uniform mired) |
| Tint | green gain 2^t, t uniform +/-0.08; all gains keep luminance |
| Sensor noise | x + sqrt(a x + b) n, a log-uniform 1e-4 to 5e-3, b log-uniform 1e-6 to 2e-4 |
| Clipping | at 1.0 |
| Tone curve | sRGB, or gamma 1.8 to 2.6, or sRGB plus an S-curve of contrast 0.8 to 1.5 |
| Saturation | 0.85 to 1.3 around Rec. 709 luminance |
| Quantisation | 8 bit |
| JPEG | 75 %: OpenCV quality 50 to 95; 25 %: none |

The label is the SH9 of the **whole** panorama rotated into the camera frame, times the same
exposure scale and white-balance gains as the image (docs/CONVENTIONS.md section 5). Blur,
vignetting, noise, clipping, the tone curve, saturation and JPEG change the picture, not the light,
so they are not in the label. `training/tests/test_synth.py` checks, on analytic environments,
that a bright lobe lands on the expected pixel with the expected label direction, that rotating the
camera rotates the label consistently, that exposure and white balance scale image and label
together, and that a white sphere lit by the label matches the photo of a uniform environment.

Panoramas are cached at 1024 x 512, divided by their median luminance and stored as float16
(clamped at 60000); the world SH is projected from the float32 image before the clamp, so a sun too
bright for float16 still contributes its full energy to the label.

## HDRI ids by split

### train (792)

```
aarfontein_dawn_2, aarfontein_dirt_road, aarfontein_dusk, aarfontein_pan, aarfontein_sunrise,
abandoned_bakery, abandoned_church, abandoned_factory_canteen_01, abandoned_factory_canteen_02,
abandoned_games_room_01, abandoned_games_room_02, abandoned_garage, abandoned_greenhouse,
abandoned_hall_01, abandoned_hopper_terminal_01, abandoned_hopper_terminal_02,
abandoned_hopper_terminal_03, abandoned_hopper_terminal_04, abandoned_parking, abandoned_pathway,
abandoned_slipway, abandoned_tank_farm_01, abandoned_tank_farm_02, abandoned_tank_farm_03,
abandoned_tank_farm_04, abandoned_tank_farm_05, abandoned_tiled_room, abandoned_workshop,
abandoned_workshop_02, acoustical_shell, adams_place_bridge, aerodynamics_workshop,
afrikaans_church_exterior, afrikaans_church_interior, aft_lounge, ahornsteig, air_museum_playground,
aircraft_workshop_01, aloe_farm_shade_house, altanka, alte_veste_station,
amphitheatre_zanzibar_fort, anniversary_lounge, approaching_storm, arboretum, aristea_wreck,
aristea_wreck_puresky, art_studio, artist_workshop, auto_service, autoshop_01, autumn_field,
autumn_field_puresky, autumn_forest_01, autumn_forest_02, autumn_forest_03, autumn_forest_04,
autumn_hill_view, autumn_hilly_field, autumn_hockey, autumn_meadow, autumn_park, aviation_museum,
aviation_museum_hill, balcony, ballawley_park, ballroom, bambanani_sunset, bamboo_tunnel,
bank_vault, barnaslingan_01, barnaslingan_02, basement_boxing_ring, bathroom, beach_cloudy_bridge,
beach_parking, belfast_farmhouse, belfast_open_field, belfast_sunset, belfast_sunset_puresky,
bell_park_dawn, bell_park_pier, bell_tower, belvedere, bergen, bethnal_green_entrance,
billiard_hall, birbeck_street_underpass, birchwood, bismarckturm, bismarckturm_hillside, blau_river,
blaubeuren_church_square, blaubeuren_hillside, blaubeuren_night, blaubeuren_outskirts,
blender_institute, blinds, blocky_photo_studio, bloem_field_sunrise, bloem_hill_01, bloem_hill_02,
bloem_hill_03, bloem_olive_house, bloem_train_track_clear, bloem_train_track_cloudy,
blouberg_sunrise_1, blouberg_sunrise_2, blue_photo_studio, boiler_room, boma, borghese_gardens,
braustuble_alley, brick_factory_01, brick_factory_02, buikslotermeerplein, burnt_warehouse,
bush_restaurant, cabin, cambridge, camdeboo_road, canary_wharf, cannon, cape_hill,
carpentry_shop_01, carpentry_shop_02, castle_zavelstein_cellar, cayley_interior, cedar_bridge,
cedar_bridge_2, cedar_bridge_sunset_1, cedar_bridge_sunset_2, champagne_castle_1, chapel_day,
chapmans_drive, charolettenbrunn_park, childrens_hospital, chinese_garden,
christmas_photo_studio_01, christmas_photo_studio_02, christmas_photo_studio_03,
christmas_photo_studio_04, christmas_photo_studio_05, christmas_photo_studio_06,
christmas_photo_studio_07, church_meeting_room, church_museum, church_stairway, cinema_hall,
cinema_lobby, circus_arena, citrus_orchard, citrus_orchard_puresky, citrus_orchard_road,
citrus_orchard_road_puresky, clarens_midday, clarens_night_01, clarens_night_02, climbing_gym,
cloud_layers, cloudy_cliffside_road, cloudy_netted_nursery, cloudy_vondelpark,
cobblestone_parish_road, cobblestone_street_night, colorful_studio, colosseum, combination_room,
comfy_cafe, concrete_tunnel, concrete_tunnel_02, construction_yard, country_club,
countrytrax_midday, courtyard, courtyard_night, cowboy_town_hall, cowboy_town_saloon,
creepy_bathroom, crossfit_gym, crystal_falls, dam_bridge, dam_road, dam_wall, dancing_hall,
dark_autumn_forest, de_balie, delta_2, derelict_airfield_01, derelict_airfield_02,
derelict_highway_midday, derelict_highway_noon, derelict_overpass, derelict_underpass,
dirt_bike_track_01, distribution_board, docklands_01, docklands_02, donkin_reserve,
drachenfels_cellar, drackenstein_quarry, drackenstein_quarry_puresky, dreifaltigkeitsberg,
dresden_moat, dresden_square, dresden_station_night, driving_school, dry_cracked_lake,
dry_hay_field, dry_meadow, dusseldorf_bridge, ehingen_hillside, ehingen_hillside_02,
eilenriede_park, emmarentia, empty_play_room, empty_workshop, en_suite, entrance_hall,
epping_forest_01, epping_forest_02, etzwihl, evening_field, evening_museum_courtyard,
evening_road_01, evening_road_01_puresky, events_hall_interior, factory_yard, farm_field,
farm_field_puresky, farm_road, farm_sunset, felsenlabyrinth, ferndale_studio_01, ferndale_studio_02,
ferndale_studio_03, ferndale_studio_04, ferndale_studio_05, ferndale_studio_06, ferndale_studio_07,
ferndale_studio_08, ferndale_studio_09, ferndale_studio_10, ferndale_studio_11, ferndale_studio_12,
fireplace, fish_eagle_hill, fish_hoek_beach, flamingo_pan, floral_tent, flower_hillside,
flower_road, forest_cave, forest_grove, forest_slope, forgotten_miniland, fort_schanskop_morning,
fouriesburg_mountain_cloudy, fouriesburg_mountain_lookout, fouriesburg_mountain_lookout_2,
fouriesburg_mountain_midday, freight_station, frozen_lake, furry_clouds, furstenstein,
future_parking, garage, garden_nook, gear_store, georgentor, glass_passage, glasshouse_interior,
glencairn_expressway, goegap, goegap_road, golden_bay, gothic_manor_01, gothic_manor_02,
graaff_reinet_groote_kerk, grasslands_sunset, graveyard_pathways, gray_pier, green_point_park,
green_sanctuary, gum_trees, gym_01, hall_of_mammals, hamburg_canal, hamburg_hbf, hangar_interior,
hanger_exterior_cloudy, hansaplatz, harties, hausdorf_clear_sky, hayloft, herkulessaulen,
highway_bridge_sunset, hilltop_construction, hilly_terrain_01, hilly_terrain_01_puresky,
historic_cloister_passage, hochsal_forest, hospital_room, hospital_room_2, hotel_rooftop_balcony,
hotel_room, illovo_beach_balcony, immenstadter_horn, indoor_pool, industrial_pipe_and_valve_01,
industrial_pipe_and_valve_02, industrial_sunset, industrial_sunset_02, industrial_sunset_02_puresky,
industrial_sunset_puresky, industrial_workshop_foundry, interior_construction, irish_institute,
je_gray_park, kart_club, kiara_1_dawn, kiara_2_sunrise, kiara_3_morning, kiara_4_mid-morning,
kiara_5_noon, kiara_6_afternoon, kiara_7_late-afternoon, kiara_8_sunset, kiara_9_dusk,
kiara_interior, killesberg_park, klippad_dawn_1, klippad_dawn_2, klippad_sunrise_1,
klippad_sunrise_2, kloetzle_blei, kloofendal_28d_misty, kloofendal_28d_misty_puresky,
kloofendal_38d_partly_cloudy, kloofendal_38d_partly_cloudy_puresky, kloofendal_43d_clear,
kloofendal_43d_clear_puresky, kloofendal_48d_partly_cloudy, kloofendal_48d_partly_cloudy_puresky,
kloofendal_misty_morning, kloofendal_misty_morning_puresky, kloofendal_overcast,
kloofendal_overcast_puresky, kloppenheim_01, kloppenheim_01_puresky, kloppenheim_02,
kloppenheim_02_puresky, kloppenheim_03, kloppenheim_03_puresky, kloppenheim_04, kloppenheim_05,
kloppenheim_05_puresky, kloppenheim_06, kloppenheim_06_puresky, kloppenheim_07,
kloppenheim_07_puresky, konzerthaus, ladybrand_heritage_house, lago_disola, lake_pier, lakes,
large_corridor, laufenurg_church, lauter_waterfall, learner_park, lebombo, leibstadt, lenong_1,
lenong_2, lenong_3, lilienstein, limehouse, little_paris_eiffel_tower, little_paris_under_tower,
lonely_road_afternoon, lonely_road_afternoon_puresky, ludwikowice_farmland, lush_dirt_path,
lythwood_room, machine_shop_01, machine_shop_02, machine_shop_03, magalies_field_sunset,
malagas_ferry_road, mall_parking_lot, mealie_road, medieval_cafe, metro_noord, metro_vijzelgracht,
minedump_flats, mirrored_hall, missile_launch_facility_01, missile_launch_facility_02,
missile_launch_facility_03, misty_dawn, misty_farm_road, misty_pines, modern_bathroom,
modern_buildings, modern_buildings_2, modern_buildings_night, modern_evening_street, monks_forest,
monkstown_castle, monochrome_studio_01, monochrome_studio_02, monochrome_studio_03,
monochrome_studio_04, monte_scherbelino, montorfano, moon_lab, moonless_golf, moonlit_golf,
mosaic_tunnel, mossy_forest, moulton_falls_train_tunnel_east, moulton_station_train_tunnel_west,
mpumalanga_veld, mpumalanga_veld_puresky, mud_road, mud_road_puresky, muddy_autumn_forest,
museum_of_ethnography, museum_of_history, museumplein, music_hall_01, music_hall_02,
nagoya_wall_path, nature_reserve_forest, near_the_river_01, near_the_river_02, netball_court,
neurathen_rock_castle, newman_cafeteria, newman_lobby, newman_locker_room, niederwihl_forest,
night_bridge, ninomaru_teien, nkuhlu, noga, noon_grass, northcliff, nowhere_road, nqweba_dawn,
old_apartments_walkway, old_bus_depot, old_depot, old_hall, old_outdoor_theater,
old_quarry_gerlingen, old_room, orbita, orlando_stadium, ostrich_road, ouchy_pier, outdoor_chapel,
outdoor_umbrellas, overcast_industrial_courtyard, overcast_soil, overcast_soil_2,
overcast_soil_puresky, palermo_sidewalk, parched_canal, park_bench, park_music_stage, park_parking,
parking_garage, partial_eclipse, passendorf_snow, pathway_morning, paul_lobe_haus, pav_studio_01,
pav_studio_02, pav_studio_03, penguin_museum, peppermint_powerplant, peppermint_powerplant_2,
petit_port, phalzer_forest_01, phone_shop, photo_studio_01, photo_studio_loft_hall,
photo_studio_london_hall, piazza_martin_lutero, pillars, pine_attic, pine_picnic, pink_sunrise,
pisztyk, pizzo_pernice, pizzo_pernice_puresky, plac_wolnosci, plains_sunset, pond,
pond_bridge_night, pool, poolbeg, potsdamer_platz, preller_drive, pretoria_gardens,
pretville_cinema, pretville_street, promenade_de_vidy, pump_house, pump_station, pylons,
quadrangle_cloudy, quadrangle_sunny, quattro_canti, qwantani, qwantani_afternoon,
qwantani_afternoon_puresky, qwantani_dawn, qwantani_dawn_puresky, qwantani_dusk_1,
qwantani_dusk_1_puresky, qwantani_dusk_2, qwantani_dusk_2_puresky, qwantani_late_afternoon,
qwantani_late_afternoon_puresky, qwantani_mid_morning, qwantani_mid_morning_puresky,
qwantani_moon_noon, qwantani_moon_noon_puresky, qwantani_moonrise, qwantani_moonrise_puresky,
qwantani_morning, qwantani_morning_puresky, qwantani_night, qwantani_night_puresky, qwantani_noon,
qwantani_noon_puresky, qwantani_patio, qwantani_puresky, qwantani_sunrise, qwantani_sunrise_puresky,
qwantani_sunset, qwantani_sunset_puresky, radkow_lake, railway_bridges, rainforest_trail, rathaus,
reading_room, red_hill_cloudy, red_hill_curve, red_hill_straight, red_wall, reichstag_1,
reinforced_concrete_01, reinforced_concrete_02, relax_inn_seaview_suite, rhodes_memorial,
river_alcove, river_rocks, river_walk_1, river_walk_2, rocky_ridge, rocky_ridge_puresky,
rogland_clear_night, rogland_moonlit_night, rogland_overcast, rogland_sunset, rolling_hills,
roof_garden, rooftop_day, rooftop_night, rooitou_park, rosendal_mountain_midmorning,
rosendal_park_sunset, rosendal_park_sunset_puresky, rosendal_plains_1, rosendal_plains_2,
rostock_arches, rostock_laage_airport, round_platform, royal_esplanade, ruckenkreuz,
rural_crossroads, rural_evening_road, rural_graffiti_tower, rural_landscape, rural_winter_roadside,
rustig_koppie, rustig_koppie_puresky, sabie_tent, safari_sunset, san_giuseppe_bridge, sandsloot,
satara_night, satara_night_no_lamps, schachen_forest, schadowplatz, school_quad,
sculpture_exhibition, scythian_tombs, scythian_tombs_2, scythian_tombs_puresky, secluded_beach,
sepulchral_chapel_basement, sepulchral_chapel_rotunda, shady_patch, shanghai_bund, shudu_lake,
signal_hill_dawn, signal_hill_sunrise, simons_town_harbour, simons_town_road, skate_park, skidpan,
skukuza_golf, skylit_garage, small_cave, small_empty_house, small_empty_room_1, small_empty_room_2,
small_empty_room_3, small_empty_room_4, small_hangar_01, small_hangar_02, small_harbor_01,
small_harbor_02, small_harbour_morning, small_harbour_sunset, small_rural_road, small_rural_road_02,
small_workshop, snow_field, snow_field_2, snow_field_2_puresky, snow_field_puresky, snowy_field,
snowy_forest, snowy_hillside, snowy_hillside_02, snowy_park_01, soliltude, solitude_interior,
solitude_night, spaichingen_hill, spiaggia_di_mondello, spooky_bamboo_morning, spree_bank,
squash_court, st_fagans_interior, st_peters_square_night, stadium_01, stadium_exterior,
sterkspruit_falls, stierberg_sunrise, stone_pines, story_studio_01, story_studio_02,
story_studio_03, story_studio_04, story_studio_05, straw_rolls_field_01, street_lamp, studio_garden,
studio_kominka_01, studio_kominka_02, studio_kominka_03, studio_kominka_04, studio_kontrast_01,
studio_kontrast_02, studio_kontrast_03, studio_kontrast_04, studio_small_01, studio_small_02,
studio_small_03, studio_small_04, studio_small_05, studio_small_06, studio_small_07,
studio_small_08, studio_small_09, studio_wizja_01, studio_wizja_02, studio_wizja_03,
stuttgart_hillside, stuttgart_suburbs, suburban_field_01, suburban_field_02, suburban_garden,
suburban_parking_area, subway_entrance, summer_stage_01, summer_stage_02, sunflowers,
sunflowers_puresky, sunny_rose_garden, sunny_vondelpark, sunset_fairway, sunset_jhbcentral,
sunset_meadow_path, syferfontein_0d_clear, syferfontein_0d_clear_puresky, syferfontein_18d_clear,
syferfontein_18d_clear_puresky, syferfontein_1d_clear, syferfontein_1d_clear_puresky,
syferfontein_6d_clear, syferfontein_6d_clear_puresky, symmetrical_garden, symmetrical_garden_02,
table_mountain_1, table_mountain_1_puresky, table_mountain_2, table_mountain_2_puresky,
tears_of_steel_bridge, teatro_massimo, teutonic_castle_moat, the_lost_city, the_sky_is_on_fire,
theater_01, theater_02, tief_etz, tiergarten, toposcope_sunset, tree_lined_driveway,
treetop_balcony, trekker_monument, tucker_wreck, turning_area, tv_studio, twilight_sunset,
umhlanga_sunrise, unfinished_office, unfinished_office_night, university_workshop, urban_alley_01,
urban_courtyard, urban_courtyard_02, urban_street_01, urban_street_02, urban_street_03,
urban_street_04, ushaka_dolphin_stadium, ushaka_sea_world_aquarium, valley_of_desolation,
vatican_road, venetian_crossroads, veranda, versveldpas, vestibule, victoria_curve_01,
victoria_curve_02, victoria_sunset, vignaioli, vignaioli_night, vintage_measuring_lab,
voortrekker_interior, vulture_hide, warm_bar, warm_reception_dinner, warm_restaurant,
warm_restaurant_night, wasteland_clouds, wasteland_clouds_puresky, waterbuck_trail, waterkloof_farm,
whale_skeleton, white_chapel, white_cliff_top, white_home_studio, wide_street_01, wide_street_02,
wildflower_field, winter_evening, winter_lake_01, winter_orchard, winter_river, winter_sky,
witsand_woolshop, wobbly_bridge, wooden_motel, woods, workshop, wrestling_gym, xiequ_yuan,
yaris_interior_garage, yellow_field, yoga_room, zavelstein, zawiszy_czarnego, zhengyang_gate,
zwinger_night
```

### val (102)

```
abandoned_waterworks, alps_field, autumn_crossing, autumn_ground, autumn_road, blue_grotto,
bryanston_park_sunrise, castel_st_angelo_roof, cave_wall, cayley_lookout, circus_maximus_1,
circus_maximus_2, cliffside, crosswalk, dalkey_view, debris_basement_corridor, decor_shop,
dikhololo_night, dikhololo_sunset, drakensberg_solitary_mountain,
drakensberg_solitary_mountain_puresky, dry_orchard_meadow, eilenriede_labyrinth, empty_warehouse_01,
geislingen_an_der_steige, german_town_street, golden_gate_hills, graffiti_shelter, hausdorf_meadow,
hay_bales, hikers_cave, homecoming_center_rooftop, industrial_wooden_attic, je_gray_02, konigsallee,
limpopo_golf_course, lot_01, lot_02, lythwood_field, lythwood_lounge, lythwood_terrace,
narrow_moonlit_road, nqweba_bridge, oberer_kuhberg, ox_bridge_morning, photo_studio_broadway_hall,
piazza_san_marco, portland_landing_pad, red_church, roofless_ruins, rural_asphalt_road,
simons_town_rocks, small_cathedral, small_cathedral_02, smelting_tower_01, smelting_tower_02,
smelting_tower_interior, snowy_cemetery, spruit_dawn, spruit_sunrise, suburban_football_field,
thatch_chapel, tiber_1, tiber_2, tiber_island, venice_dawn_1, venice_dawn_2, venice_sunrise,
venice_sunset, viale_giuseppe_garibaldi, whipple_creek_gazebo, whipple_creek_regional_park_01,
whipple_creek_regional_park_04, wooden_studio_01, wooden_studio_02, wooden_studio_03,
wooden_studio_04, wooden_studio_05, wooden_studio_06, wooden_studio_07, wooden_studio_08,
wooden_studio_09, wooden_studio_10, wooden_studio_11, wooden_studio_12, wooden_studio_13,
wooden_studio_14, wooden_studio_15, wooden_studio_16, wooden_studio_17, wooden_studio_18,
wooden_studio_19, zwartkops_curve_afternoon, zwartkops_curve_morning, zwartkops_curve_sunset,
zwartkops_pit, zwartkops_start_afternoon, zwartkops_start_morning, zwartkops_start_sunset,
zwartkops_straight_afternoon, zwartkops_straight_morning, zwartkops_straight_sunset
```

### test (103)

```
abandoned_construction, between_bridges, binnenalster, blue_lagoon, blue_lagoon_night,
brown_photostudio_01, brown_photostudio_02, brown_photostudio_03, brown_photostudio_04,
brown_photostudio_05, brown_photostudio_06, brown_photostudio_07, cyclorama_hard_light, dry_field,
evening_meadow, farmland_overcast, gamrig, golf_course_sunrise, greenwich_park, greenwich_park_02,
greenwich_park_03, gym_entrance, hall_of_finfish, harties_cliff_view, harvest, hochsal_field,
horn-koppe_snow, horn-koppe_spring, lakeside, lakeside_dawn, lakeside_night, lakeside_sunrise, lapa,
leadenhall_market, lookout, marry_hall, meadow, meadow_2, monbachtal_riverbank, mutianyu,
neon_photostudio, neuer_zollhof, old_tree_in_city_park, outdoor_workshop, palermo_park,
palermo_square, pedestrian_overpass, pergola_walkway, piazza_bologni, poly_haven_studio,
provence_studio, quarry_01, quarry_01_puresky, quarry_02, quarry_03, quarry_04, quarry_04_puresky,
quarry_cloudy, railway_bridge_02, residential_garden, resting_place, resting_place_2, rotes_rathaus,
school_hall, shanghai_riverside, short_tunnel, sisulu, snowy_forest_path_01, snowy_forest_path_02,
steinbach_field, stone_alley, stone_alley_02, stone_alley_03, storeroom, stream,
studio_country_hall, suburban_soccer_park, sundowner_deck, sundowner_overlook, sunny_country_road,
sunset_forest, sunset_in_the_chalk_quarry, surgery, teufelsberg_ground_1, teufelsberg_ground_2,
teufelsberg_inner, teufelsberg_lookout, teufelsberg_roof, ticknock_01, ticknock_02, ticknock_03,
ticknock_04, ulmer_muenster, under_bridge, veld_fire, white_studio_01, white_studio_02,
white_studio_03, white_studio_04, white_studio_05, white_studio_06, wooden_lounge, xanderklinge
```

### ood (48)

```
DayEnvironmentHDRI002, DayEnvironmentHDRI004, DayEnvironmentHDRI005, DayEnvironmentHDRI007,
DayEnvironmentHDRI009, DayEnvironmentHDRI013, DayEnvironmentHDRI015, DayEnvironmentHDRI017,
DayEnvironmentHDRI039, DayEnvironmentHDRI040, DayEnvironmentHDRI046, DayEnvironmentHDRI047,
DayEnvironmentHDRI051, DayEnvironmentHDRI059, DayEnvironmentHDRI064, DayEnvironmentHDRI066,
DayEnvironmentHDRI071, DayEnvironmentHDRI077, DayEnvironmentHDRI082, DayEnvironmentHDRI107,
EveningEnvironmentHDRI001, EveningEnvironmentHDRI003, EveningEnvironmentHDRI004,
IndoorEnvironmentHDRI001, IndoorEnvironmentHDRI002, IndoorEnvironmentHDRI003,
IndoorEnvironmentHDRI004, IndoorEnvironmentHDRI005, IndoorEnvironmentHDRI006,
IndoorEnvironmentHDRI007, IndoorEnvironmentHDRI008, IndoorEnvironmentHDRI009,
IndoorEnvironmentHDRI010, IndoorEnvironmentHDRI011, IndoorEnvironmentHDRI012,
IndoorEnvironmentHDRI013, IndoorEnvironmentHDRI014, IndoorEnvironmentHDRI015,
IndoorEnvironmentHDRI016, IndoorEnvironmentHDRI017, IndoorEnvironmentHDRI018,
IndoorEnvironmentHDRI019, IndoorEnvironmentHDRI020, IndoorEnvironmentHDRI021,
IndoorEnvironmentHDRI022, IndoorEnvironmentHDRI023, NightEnvironmentHDRI002, NightEnvironmentHDRI007
```
