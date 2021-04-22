import requests
import aiohttp
import asyncio
import utm
import cv2
import numpy as np

# https://www.icgc.cat/Administracio-i-empresa/Serveis/Geoinformacio-en-linia-Geoserveis

def dataAugmentationFunction(image, mask):
    result = [image]
    masks = [mask]

    # Canviar color ---------------------------

    # Afegir blau (simula nit)
    for i in [(x+1)/10.0+1 for x in range(8)]:
        newImage = image.copy()
        newImage[:,:,0] = np.clip((newImage[:,:,0]*i), 0, 255).astype(np.uint8)
        result.append(newImage)
        masks.append(mask.copy()) # La mascara no canvia
        cv2.imshow("Image", newImage)
        cv2.waitKey()

    # Enfosquir
    for im in result.copy():
        for i in [0.5 + x/10.0 for x in range(5)]:
            newImage = im.copy()
            newImage = np.clip((newImage*i), 0, 255).astype(np.uint8)
            result.append(newImage)
            masks.append(mask.copy()) # La mascara no canvia
            cv2.imshow("Image", newImage)
            cv2.waitKey()

    # Zoom i Translacions ---------------------------
    # for im, mask in zip(result.copy(), masks.copy()):



class ICGCService:
    def __init__(self):
        self.imagesUrl = "https://geoserveis.icgc.cat/icc_ortohistorica/wms/service"
        self.altitudeUrl = "http://geoserveis.icgc.cat/icgc_mdt2m/wms/service"


    async def getSatImage(self, minLat, minLon, maxLat, maxLon, width=520, height=520, layer="orto25c2016"):
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
        # r = requests.get(self.imagesUrl, params=params)
        # image = cv2.imdecode(np.frombuffer(r.content, np.uint8), -1)
        # return image

        session = aiohttp.ClientSession()
        content = b''
        async with session.get(self.imagesUrl, params=params) as resp:
            content = await resp.content.read()
        image = cv2.imdecode(np.frombuffer(content, np.uint8), -1)
        await session.close()
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
    image = asyncio.run(service.getSatImage(41.626854692115955, 2.250532669633619, 41.627898950089, 2.2514200465686196, layer="orto25c2016"))
    cv2.imshow("Image", image)
    # cv2.waitKey()

    altImage = service.getAltImage(41.73015520828032, 2.29140190952102, 41.829572811017734, 2.4619104037745116)
    print(altImage.min(), altImage.max())
    cv2.imshow("Image2", altImage)
    cv2.waitKey()

    
