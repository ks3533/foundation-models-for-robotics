from Code.robocasa_env.main import Controller

for _ in range(20):
    controller = Controller()
    controller.start()

    controller.open_door()
    controller.open_door()

    controller.grip_object_from_above("obj")

    controller.place_object_at_destination("obj", "container")

    controller.close_door()

    controller.press_button()

    open("correct_behavior.log", "a").write(f"{controller.env.objects['obj'].root.attrib['model']} "
                                            f"{'passed' if controller.check_successful() else 'failed'}\n")
    controller.stop()
    print("finished simulation")

    controller = Controller()
    controller.start()

    controller.open_door()

    controller.grip_object_from_above("obj")
    controller.grip_object_from_above("obj")

    controller.place_object_at_destination("obj", "container")

    controller.close_door()

    controller.press_button()

    open("correct_behavior.log", "a").write(f"{controller.env.objects['obj'].root.attrib['model']} "
                                            f"{'passed' if controller.check_successful() else 'failed'}\n")
    controller.stop()
    print("finished simulation")

    controller = Controller()
    controller.start()

    controller.grip_object_from_above("obj")

    controller.open_door()

    controller.grip_object_from_above("obj")

    controller.place_object_at_destination("obj", "container")

    controller.close_door()

    controller.press_button()

    open("correct_behavior.log", "a").write(f"{controller.env.objects['obj'].root.attrib['model']} "
                                            f"{'passed' if controller.check_successful() else 'failed'}\n")
    controller.stop()
    print("finished simulation")
