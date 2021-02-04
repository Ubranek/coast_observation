class TrackableObject:
	def __init__(self, objectID, centroid, class_name):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.current_centroid = centroid
		self.class_name = class_name

		#for saving pairs zone.id: dir
		#where dir is 1 if object was noticed in zone, and 0 if not
		self.zone_checks = dict()

	# 1 - entry, -1 - exit, 0 no changes
	def get_direction(self, zone_id, event_type, new_state):
		zone_id = "{}_{}".format(event_type, zone_id)
		old_zone_state = self.zone_checks.get(zone_id, None)
		if old_zone_state is not None and old_zone_state != new_state:
			self.zone_checks[zone_id] = new_state
			return -1 if new_state == 0 else  1
		elif old_zone_state is None:
			self.zone_checks[zone_id] = new_state
			return 1 if new_state == 1 else 0

		return 0

	@property
	def int_id(self):
		return int(self.objectID.split("_")[1])