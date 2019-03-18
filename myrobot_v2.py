import sc2
from sc2 import run_game, maps, Race, Difficulty, Result
from sc2.player import Bot, Computer
from sc2 import position
from sc2.constants import *
import cv2
import math
import random
import numpy as np
import os
import time
# import keras

#os.environ["SC2PATH"] = '/starcraftstuff/StarCraftII/'
os.chdir('C:/Users/DMoney/Desktop/Programming/Python/python-sc2-master')
HEADLESS = False

class DanBot(sc2.BotAI):
    def __init__(self, title = 1, use_model=False, use_model2 = False):
        self.MAX_WORKERS = 80
        self.do_something_after = 40
        self.use_model = use_model
        self.use_model2 = use_model2
        self.K = 8
        self.scouts_and_spots = {}
        self.title = title

        self.train_data = []
        #####
        # if self.use_model:
        #     print("USING MODEL!")
        #     self.model = keras.models.load_model("BasicCNN-30-epochs-0.0001-LR-4.2") #Attack brain
        # if self.use_model2:
        #     print("USING MODEL2!")
        #     #self.model2 = keras.models.load_model('') #


    def on_end(self, game_result):
        print('--- on_end called ---')
        print(game_result, self.use_model)

        with open("gameout-random-vs-medium.txt","a") as f:
            if self.use_model:
                f.write("Model {} - {}\n".format(game_result, int(time.time())))
            else:
                f.write("Random {} - {}\n".format(game_result, int(time.time())))

        if game_result == Result.Victory:
            np.save("train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data))

    async def on_step(self, iteration):
        self.iteration = iteration
        self.game_time = (self.state.game_loop/22.4) #game time in seconds
        await self.scout()
        # await self.attack()
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.intel()
        await self.macro_brain()

        for nexus in self.units(NEXUS):
            if not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                abilities = await self.get_available_abilities(nexus)
                if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities:
                    await self.do(nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, nexus))

# These will go in Brain 2
        # await self.expand()
        # await self.tech_up()
        # await self.build_offensive_force()


    def random_location_variance(self, location):
        x = location[0]
        y = location[1]

        #  FIXED THIS
        x += random.randrange(-5,5)
        y += random.randrange(-5,5)

        if x < 0:
            print("x below")
            x = 0
        if y < 0:
            print("y below")
            y = 0
        if x > self.game_info.map_size[0]:
            print("x above")
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            print("y above")
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x,y)))

        return go_to

    async def scout(self):

        self.expand_dis_dir = {}

        for el in self.expansion_locations:
            distance_to_enemy_start = el.distance_to(self.enemy_start_locations[0])
            #print(distance_to_enemy_start)
            self.expand_dis_dir[distance_to_enemy_start] = el

        self.ordered_exp_distances = sorted(k for k in self.expand_dis_dir)

        existing_ids = [unit.tag for unit in self.units]
        # removing of scouts that are actually dead now.
        to_be_removed = []
        for noted_scout in self.scouts_and_spots:
            if noted_scout not in existing_ids:
                to_be_removed.append(noted_scout)

        for scout in to_be_removed:
            del self.scouts_and_spots[scout]

        if len(self.units(ROBOTICSFACILITY).ready) == 0:
            unit_type = PROBE
            unit_limit = 1
        else:
            unit_type = OBSERVER
            unit_limit = 15

        assign_scout = True

        if unit_type == PROBE:
            for unit in self.units(PROBE):
                if unit.tag in self.scouts_and_spots:
                    assign_scout = False

        if assign_scout:
            if len(self.units(unit_type).idle) > 0:
                for obs in self.units(unit_type).idle[:unit_limit]:
                    if obs.tag not in self.scouts_and_spots:
                        for dist in self.ordered_exp_distances:
                            try:
                                location = next(value for key, value in self.expand_dis_dir.items() if key == dist)
                                # DICT {UNIT_ID:LOCATION}
                                active_locations = [self.scouts_and_spots[k] for k in self.scouts_and_spots]

                                if location not in active_locations:
                                    if unit_type == PROBE:
                                        for unit in self.units(PROBE):
                                            if unit.tag in self.scouts_and_spots:
                                                continue

                                    await self.do(obs.move(location))
                                    self.scouts_and_spots[obs.tag] = location
                                    break
                            except Exception as e:
                                pass

        for obs in self.units(unit_type):
            if obs.tag in self.scouts_and_spots:
                if obs in [probe for probe in self.units(PROBE)]:
                    await self.do(obs.move(self.random_location_variance(self.scouts_and_spots[obs.tag])))

    async def intel(self):

        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        for unit in self.units().ready:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*8), (0, 125, 0), math.ceil(int(unit.radius*0.5)))

        for unit in self.known_enemy_units:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*8), (0, 0, 125), math.ceil(int(unit.radius*0.5)))

        try:
            line_max = 50
            mineral_ratio = self.minerals / 1500
            if mineral_ratio > 1.0:
                mineral_ratio = 1.0

            vespene_ratio = self.vespene / 1500
            if vespene_ratio > 1.0:
                vespene_ratio = 1.0

            population_ratio = self.supply_left / self.supply_cap
            if population_ratio > 1.0:
                population_ratio = 1.0

            plausible_supply = self.supply_cap / 200.0

            worker_weight = len(self.units(PROBE)) / (self.supply_cap-self.supply_left)
            if worker_weight > 1.0:
                worker_weight = 1.0

            cv2.line(game_data, (0, 19), (int(line_max*worker_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
            cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
            cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
            cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
            cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500
        except Exception as e:
            print(str(e))

        # flip horizontally to make our final fix in visual representation:

        self.flipped = cv2.flip(game_data, 0)
        resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
        if not HEADLESS:
            cv2.imshow(str(self.title), resized)
            cv2.waitKey(1)

    async def build_workers(self):
        if (len(self.units(NEXUS)) * 30) > len(self.units(PROBE)) and len(self.units(PROBE)) < self.MAX_WORKERS:
            for nexus in self.units(NEXUS).ready.noqueue:
                if self.can_afford(PROBE):
                    await self.do(nexus.train(PROBE))

    async def build_pylons(self):
        if self.supply_left < 5 and not self.already_pending(PYLON):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=nexuses.first)

    async def build_assimilators(self):
        for nexus in self.units(NEXUS).ready:
            vaspenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vaspene in vaspenes:
                if not self.can_afford(ASSIMILATOR):
                    break
                worker = self.select_build_worker(vaspene.position)
                if worker is None:
                    break
                if not self.units(ASSIMILATOR).closer_than(1.0, vaspene).exists:
                    await self.do(worker.build(ASSIMILATOR, vaspene))


    async def expand(self):
        try:
            if self.can_afford(NEXUS) and not self.already_pending(NEXUS):
                await self.expand_now()
        except Exception as e:
            print(str(e))


    async def tech_up(self):

        possible_buildings = [GATEWAY, CYBERNETICSCORE, STARGATE, 
                              ROBOTICSFACILITY, TWILIGHTCOUNCIL, FLEETBEACON, ROBOTICSBAY, TEMPLARARCHIVE, DARKSHRINE]

        progress = 0

        for building in possible_buildings:
            if self.units(building).exists:
                progress = possible_buildings.index(building) + 1

        if progress >= len(possible_buildings):
            progress = random.randrange(0,4)
            if progress == 2:
                progress = 1

        # possible_units = [ZEALOT, SENTRY, STALKER, HIGHTEMPLAR, DARKTEMPLAR, ADEPT, PHOENIX, ORACLE, VOIDRAY, TEMPEST, CARRIER, OBSERVER, WARPPRISM, IMMORTAL, COLOSSUS, DISRUPTOR, MOTHERSHIP]
        # if self.units(DARKSHRINE).exists:
        #     pass
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            if self.can_afford(possible_buildings[progress]) and not self.already_pending(possible_buildings[progress]) and not self.already_pending(possible_buildings[progress-1]):
                await self.build(possible_buildings[progress], near = pylon)
            

        # if self.units(PYLON).ready.exists:
        #     pylon = self.units(PYLON).ready.random
        #     num_times_teched_this_pass = 0

        #     while num_times_teched_this_pass  < 1:
        #         try:
        #             random_tech_choice = possible_buildings[random.randrange(0,len(possible_buildings))]
        #             print("I'm gonna build a " + random_tech_choice)
        #                 if self.can_afford(random_tech_choice) and not self.already_pending(random_tech_choice):
        #                     await self.build(random_tech_choice, near = pylon)
        #         num_times_teched_this_pass += 1

        #     except:
        #         pass

    async def tech_side(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            await self.build(GATEWAY, near = pylon)

    async def build_mans(self):
        gateway_mans = [ZEALOT, STALKER]#, SENTRY, ADEPT]
        stargate_mans = [VOIDRAY,TEMPEST,CARRIER]
        robotics_mans = [IMMORTAL,OBSERVER,COLOSSUS]

        for gw in self.units(GATEWAY).ready.noqueue:
            if self.units(CYBERNETICSCORE).exists:
                random_man = random.randrange(0,len(gateway_mans))
                if self.can_afford(gateway_mans[random_man]) and self.supply_left > 0:
                    await self.do(gw.train(gateway_mans[random_man]))
            else:
                if self.can_afford(gateway_mans[0]) and self.supply_left > 0:
                    await self.do(gw.train(gateway_mans[0]))

        for sg in self.units(STARGATE).ready.noqueue:
            if self.units(FLEETBEACON).exists:
                random_man = random.randrange(0,len(stargate_mans))
                if self.can_afford(stargate_mans[random_man]) and self.supply_left > 0:
                    await self.do(sg.train(stargate_mans[random_man]))
            else:
                if self.can_afford(stargate_mans[0]) and self.supply_left > 0:
                    await self.do(sg.train(stargate_mans[0]))

        for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
            if self.units(ROBOTICSBAY).exists:
                random_man = random.randrange(0,len(robotics_mans))
                if self.can_afford(robotics_mans[random_man]) and self.supply_left > 0:
                    await self.do(rf.train(robotics_mans[random_man]))
            else:
                random_man = random.randrange(0,len(robotics_mans)-1)
                if self.can_afford(robotics_mans[random_man]) and self.supply_left > 0:
                    await self.do(rf.train(robotics_mans[random_man]))

        # for gw in self.units(GATEWAY).ready.noqueue:
        #     random_man = random.randrange(0,len(gateway_mans))
        #     if self.can_afford(random_man) and self.supply_left > 0:
        #         await self.do(gw.train(random_man))

        # for sg in self.units(STARGATE).ready.noqueue:
        #     random_man = random.randrange(0,len(stargate_mans))
        #     if self.can_afford(random_man) and self.supply_left > 0:
        #         await self.do(sg.train(random_man))

        # for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
        #     random_man = random.randrange(0,len(robotics_mans))
        #     if self.can_afford(random_man) and self.supply_left > 0:
        #         await self.do(rf.tain(random_man))

    async def defend_nexus(self):
        possible_units = [ZEALOT, SENTRY, STALKER, HIGHTEMPLAR, DARKTEMPLAR, ADEPT, PHOENIX, ORACLE, VOIDRAY, TEMPEST, CARRIER, OBSERVER, WARPPRISM, IMMORTAL, COLOSSUS, DISRUPTOR, MOTHERSHIP]
        for unit in possible_units:
            if len(self.known_enemy_units) > 0:
                target = self.known_enemy_units.closest_to(random.choice(self.units(NEXUS)))
                for u in self.units(unit).idle:
                    await self.do(u.attack(target))

    async def attack_known_enemy_structure(self):
        possible_units = [ZEALOT, SENTRY, STALKER, HIGHTEMPLAR, DARKTEMPLAR, ADEPT, PHOENIX, ORACLE, VOIDRAY, TEMPEST, CARRIER, OBSERVER, WARPPRISM, IMMORTAL, COLOSSUS, DISRUPTOR, MOTHERSHIP]
        for unit in possible_units:
            if len(self.known_enemy_structures) > 0:
                target = random.choice(self.known_enemy_structures)
                for u in self.units(unit).idle:
                    await self.do(u.attack(target))

    async def attack_known_enemy_unit(self):
        possible_units = [ZEALOT, SENTRY, STALKER, HIGHTEMPLAR, DARKTEMPLAR, ADEPT, PHOENIX, ORACLE, VOIDRAY, TEMPEST, CARRIER, OBSERVER, WARPPRISM, IMMORTAL, COLOSSUS, DISRUPTOR, MOTHERSHIP]
        for unit in possible_units:
            if len(self.known_enemy_units) > 0:
                target = self.known_enemy_units.closest_to(random.choice(self.units(NEXUS)))
                for u in self.units(unit).idle:
                    await self.do(u.attack(target.position))
            elif len(self.known_enemy_structures) > 0:
                target = random.choice(self.known_enemy_structures)
                for u in self.units(unit).idle:
                    await self.do(u.attack(target.position))
            else:
                for u in self.units(unit).idle:
                    await self.do(u.attack(self.enemy_start_locations[0]))


    async def macro_brain(self):
        choice_dict = {0: "Doing nothing!",
                       1: "Expanding!",
                       2: "Teching!",
                       3: "Building mans!",
                       4: "More gateways!",
                       5: "Attacking enemy!",
                       #6: "Checking money!"
                       }

        if self.game_time > self.do_something_after:
            if self.use_model2:
                prediction = self.model2.predict([self.flipped.reshape([-1, 176, 200, 3])])
                choice = np.argmax(prediction[0])

                print("Macro brain says #{}:{}".format(choice, choice_dict[choice]))

            else:
                # choices = [0,1,2,3,3,4,5]
                choice = random.randrange(0,len(choice_dict))
            if choice == 0:
                print("Macro brain says #{}:{}".format(choice, choice_dict[choice]))
                self.do_something_after = self.game_time + self.K
            elif choice == 1:
                print("Macro brain says #{}:{}".format(choice, choice_dict[choice]))
                self.do_something_after = self.game_time + self.K
                await self.expand()
            elif choice == 2:
                print("Macro brain says #{}:{}".format(choice, choice_dict[choice]))
                self.do_something_after = self.game_time + self.K
                await self.tech_up()
            elif choice == 3:
                print("Macro brain says #{}:{}".format(choice, choice_dict[choice]))
                self.do_something_after = self.game_time + self.K
                await self.build_mans()
            elif choice == 4:
                print("Macro brain says #{}:{}".format(choice, choice_dict[choice]))
                self.do_something_after = self.game_time + self.K
                await self.tech_side()
            elif choice == 5:
                print("Macro brain says #{}:{}".format(choice, choice_dict[choice]))
                self.do_something_after = self.game_time + self.K
                await self.attack_known_enemy_unit()
            # elif choice == 6:
            #     print("Macro brain says #{}:{}".format(choice, choice_dict[choice]))
            #     self.do_something_after = self.game_time + self.K                

            y = np.zeros(6)
            y[choice] = 1
            #print(y)
            self.train_data.append([y,self.flipped])





    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]

    # async def attack(self):
    #     possible_units = [ZEALOT, SENTRY, STALKER, HIGHTEMPLAR, DARKTEMPLAR, ADEPT, PHOENIX, ORACLE, VOIDRAY, TEMPEST, CARRIER, OBSERVER, WARPPRISM, IMMORTAL, COLOSSUS, DISRUPTOR, MOTHERSHIP]


    #     if sum([len(self.units(unit).idle) for unit in possible_units]) > 0:

    #         target = False
    #         if self.game_time > self.do_something_after:
    #             if self.use_model:
    #                 prediction = self.model.predict([self.flipped.reshape([-1, 176, 200, 3])], batch_size = 16)
    #                 choice = np.argmax(prediction[0])
    #                 #print('prediction: ',choice)

    #                 choice_dict = {0: "No Attack!",
    #                                1: "Attack close to our nexus!",
    #                                2: "Attack Enemy Structure!",
    #                                3: "Attack Enemy Start!"}

    #                 print("Attack brain says #{}:{}".format(choice, choice_dict[choice]))

    #             else:
    #                 choice = random.randrange(0, 4)

    #             if choice == 0:
    #                 # no attack
    #                 #wait = random.randrange(20,165)
    #                 self.do_something_after = self.game_time + self.K

    #             elif choice == 1:
    #                 #attack_unit_closest_nexus
    #                 if len(self.known_enemy_units) > 0:
    #                     target = self.known_enemy_units.closest_to(random.choice(self.units(NEXUS)))

    #             elif choice == 2:
    #                 #attack enemy structures
    #                 if len(self.known_enemy_structures) > 0:
    #                     target = random.choice(self.known_enemy_structures)

    #             elif choice == 3:
    #                 #attack_enemy_start
    #                 target = self.enemy_start_locations[0]

    #             if target:
    #                 for UNIT in possible_units:
    #                     for s in self.units(UNIT).idle:
    #                         await self.do(s.attack(self.find_target(self.state)))

                # y = np.zeros(4)
                # y[choice] = 1
                # #print(y)
                # self.train_data.append([y,self.flipped])

            #print(len(self.train_data))

for i in range(444):
    print("Game ", i+1)

    run_game(maps.get("AbyssalReefLE"), [
        Bot(Race.Protoss, DanBot(title = 1, use_model=True,use_model2 = False)),
        Computer(Race.Terran, Difficulty.Medium),
        ], realtime=False)
