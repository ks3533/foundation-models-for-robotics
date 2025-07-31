from Code.robocasa_env.main import Controller

controller = Controller()
controller.start()

controller.open_door()

controller.grip_object_from_above("obj")

controller.place_object_at_destination("obj", "container")

controller.close_door()

controller.press_button()

# controller.stop()
print("finished simulation")
