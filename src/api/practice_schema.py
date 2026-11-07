from src.api.schemas import PredictionRequest

history = []

for cycle in range(1, 21):
    observation = {
        "time_in_cycles": cycle,
        "operational_setting_1": 0.0,
        "operational_setting_2": 0.0,
        "operational_setting_3": 100.0,
    }

    for sensor in range(1, 22):
        observation[f"sensor_measurement_{sensor}"] = 1.0

    history.append(observation)

data = {
    "engine_id": 42,
    "history": history,
}

request = PredictionRequest.model_validate(data) #We inherit model_validate() from BaseModel; it checks data against our class’s rules and returns an object of our class.

print("Validation succeeded")
print("Engine:", request.engine_id)
print("Observation count:", len(request.history))
print("First cycle:", request.history[0].time_in_cycles)