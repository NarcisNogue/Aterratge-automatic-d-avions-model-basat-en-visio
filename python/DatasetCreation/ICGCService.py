import requests
import utm
import cv2
import numpy as np

# https://www.icgc.cat/Administracio-i-empresa/Serveis/Geoinformacio-en-linia-Geoserveis

class ICGCService:
    def __init__(self):
        self.imagesUrl = "https://geoserveis.icgc.cat/icc_ortohistorica/wms/service"
        self.altitudeUrl = "http://geoserveis.icgc.cat/icgc_mdt2m/wms/service"


    def getSatImage(self, minLat, minLon, maxLat, maxLon, width=520, height=520, layer="orto25c2016"):
        minCords = ",".join(str(i) for i in utm.from_latlon(minLat, minLon)[0:2])
        maxCords = ",".join(str(i) for i in utm.from_latlon(maxLat, maxLon)[0:2]) #0xFFFFFF&TRANSPARENT=TRUE&EXCEPTION=INIMAGE
        params = {
            "REQUEST" : "GetMap",
            "VERSION" : "1.1.0",
            "SERVICE" : "WMS",
            "SRS"     : "EPSG:25831",
            "BBOX"    : ",".join([minCords, maxCords]),
            "WIDTH"   : str(width),
            "HEIGHT"  : str(height),
            "LAYERS"  : layer,
            "STYLES"  : "",
            "FORMAT"  : "JPEG",
            "BGCOLOR" : "0xFFFFFF",
            "TRANSPARENT" : "TRUE",
            "EXCEPTION"   : "INIMAGE"
        }

        # REQUEST
        r = requests.get(self.imagesUrl, params=params)
        image = cv2.imdecode(np.frombuffer(r.content, np.uint8), -1)
        return image

# https://geoserveis.icgc.cat/icgc_mdt2m/wms/service?REQUEST=GetMap&SERVICE=WMS&VERSION=1.3.0
# &LAYERS=MET2m&STYLES=&FORMAT=image/png&BGCOLOR=0xFFFFFF&TRANSPARENT=FALSE&CRS=EPSG:25831
# BBOX=369265.002588546,4658150.50009494,409113.574325388,4698925.7828024&WIDTH=817&HEIGHT=836


    def getAltImage(self, minLat, minLon, maxLat, maxLon, width=520, height=520, layer="MET2m"):
        #  0  --> 6,1m
        # 255 --> 3142,9m 
        minCords = ",".join(str(i) for i in utm.from_latlon(minLat, minLon)[0:2])
        maxCords = ",".join(str(i) for i in utm.from_latlon(maxLat, maxLon)[0:2])
        params = {
            "REQUEST" : "GetMap",
            "VERSION" : "1.3.0",
            "SERVICE" : "WMS",
            "SRS"     : "EPSG:25831",
            "BBOX"    : ",".join([minCords, maxCords]),
            "WIDTH"   : str(width),
            "HEIGHT"  : str(height),
            "LAYERS"  : layer,
            "STYLES"  : "",
            "CRS"     : "EPSG:25831",
            "FORMAT"  : "image/png",
            "BGCOLOR" : "0xFFFFFF",
            "TRANSPARENT" : "FALSE"
        }

        # REQUEST
        r = requests.get(self.altitudeUrl, params=params)
        image = cv2.imdecode(np.frombuffer(r.content, np.uint8), -1)
        return image

# TESTS
if __name__ == "__main__":
    # 41.626854692115955, 2.250532669633619, 41.627898950089, 2.2514200465686196
    service = ICGCService()
    image = service.getSatImage(41.626854692115955, 2.250532669633619, 41.627898950089, 2.2514200465686196, layer="orto25c2016")
    cv2.imshow("Image", image)
    # cv2.waitKey()

    altImage = service.getAltImage(41.73015520828032, 2.29140190952102, 41.829572811017734, 2.4619104037745116)
    print(altImage.min(), altImage.max())
    cv2.imshow("Image2", altImage)
    cv2.waitKey()

    
