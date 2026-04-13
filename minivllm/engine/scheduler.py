# engine/scheduler.py

class Scheduler:
    def __init__(self):
        self.waiting_queue = []
        self.running = []
    
    def add_request(self, request):
        self.waiting_queue.append(request)
    
    def step(self):
        while self.waiting_queue:
            self.running.append(self.waiting_queue.pop(0))
        return self.running
    