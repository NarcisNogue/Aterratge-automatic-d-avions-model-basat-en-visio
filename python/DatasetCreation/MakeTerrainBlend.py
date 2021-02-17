import bpy
import ICGCService

# print all objects
plane = bpy.data.objects["Plane"]

plane.location = [0,0,0]

service = ICGCService()

altImage = service.getAltImage(41.73015520828032, 2.29140190952102, 41.829572811017734, 2.4619104037745116)