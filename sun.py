import time
import subprocess
import sys

import cv2
import numpy
from numpy import sin, cos, tan, pi, arccos, arctan


def get_sun_pos(day, hour, min, sec, lat, long, tz):
    """
    Compute the position of the sun in the sky
    Source: https://gml.noaa.gov/grad/solcalc/solareqns.PDF with modifications
    based on the source code of https://gml.noaa.gov/grad/solcalc/azel.html
    The equation of time is taken from the corresponding Wikipedia page
    :param day: The day of the year, where January 1st is 0
    :param hour: hour in UTC
    :param min: minute in UTC
    :param sec: second in UTC
    :param lat: The latitude of the observer in degrees
    :param long: The longitude of the observer in degrees
    :return: (azimuth angle, zenith angle) in radians
    """
    gamma = 2*pi/365 * (day + (hour - 12)/24)

    eot_n = 2*pi / 365.24
    eot_a = eot_n*(day + 10)
    eot_b = eot_a + 2*0.0167*sin(eot_n*(day - 2))
    eot_c = (eot_a - arctan(tan(eot_b) / 0.91747714052))/pi

    # eqtime = 229.18 * (0.000075 + 0.001868*cos(gamma) - 0.032077*sin(gamma) - 0.014615*cos(2*gamma) - 0.040849*sin(2*gamma))
    eqtime = 720*(eot_c - round(eot_c))
    fy = day + hour/24
    decl = 0.006918 - 0.399912*cos(gamma) + 0.070257*sin(gamma) - 0.006758*cos(2*gamma) + 0.000907*sin(2*gamma) - 0.002697*cos(3*gamma) + 0.00148*sin(3*gamma)
    # decl = arcsin(0.39779 * cos(2*pi/365.24 * (fy+10) + 2*0.0167*sin(2*pi/365.24 * (fy-2))))
    ha = ((hour*60 + min + sec/60 + eqtime + 4*long - tz*60) / 4 - 180) * pi/180
    if ha < -pi: ha += 2*pi

    lat = pi/180 * lat
    zenith = arccos(sin(lat)*sin(decl) + cos(lat)*cos(decl)*cos(ha))
    azimuth = pi - arccos((sin(lat)*cos(zenith) - sin(decl)) / (cos(lat)*sin(zenith)))
    if ha > 0: azimuth = 2*pi - azimuth
    return azimuth, zenith


def sphere2xy(a, z):
    x = int(param["fullWidth"] / 360 * (a * 180/pi - param["heading"]) + width//2)
    return (x + param["xstart"]) % param["fullWidth"] - param["xstart"], int(z / pi * param["fullHeight"] - param["ystart"])


def extract_exif(file):
    o = subprocess.check_output(["exiftool", "-G", "-a", "-xmp:all", file]).decode()
    return {r.split(":")[0][5:].strip(): r.split(":")[1].strip() for r in o.split("\n") if ":" in r}


if __name__ == "__main__":
    exif = extract_exif(sys.argv[1])
    param = {
        "fullHeight": int(exif["Full Pano Height Pixels"]),
        "fullWidth": int(exif["Full Pano Width Pixels"]),
        "ystart": int(exif["Cropped Area Top Pixels"]),
        "xstart": int(exif["Cropped Area Left Pixels"]),
        "heading": float(exif["Pose Heading Degrees"])
    }
    lat = float(sys.argv[2])
    long = float(sys.argv[3])

    dt = 15
    p_len = 24*60//dt
    positions = numpy.zeros((p_len, 2))




    img = cv2.imread(sys.argv[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]

    mask = numpy.zeros((3, height+2, width+2), dtype=numpy.uint8)

    sobel_x = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3))
    sobel_y = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3))
    sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 1)
    sobel = cv2.threshold(cv2.dilate(sobel, numpy.ones((25, 25))), 25, 255, cv2.THRESH_BINARY)[1]

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


    def clicked(event, x, y, flags, param):
        global mask, sobel, img, positions, width, height
        if event == cv2.EVENT_LBUTTONDOWN and not sobel[y, x, 0]:
            cv2.floodFill(sobel, mask[2], (x, y), 255)
            t_mask = mask.transpose([1, 2, 0])
            cv2.destroyAllWindows()
            img_c = cv2.addWeighted(img, 0.7, (t_mask*255)[1:-1, 1:-1], 0.3, 1)
            cv2.imwrite("/tmp/sobel.jpg", sobel)
            cv2.imshow("Panorama", cv2.vconcat([img_c, sobel]))
            cv2.setMouseCallback("Panorama", clicked)

    sobel = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("/tmp/sobel.jpg", sobel)
    cv2.imshow("Panorama", cv2.vconcat([img, sobel]))
    cv2.setMouseCallback("Panorama", clicked)
    cv2.waitKey(0)

    # area = find_area(sobel, (width//2, height//2))
    # cv2.line(img, (width//2, 0), (width//2, height), (0, 0, 255), 10)
    east = sphere2xy(pi/2, 0)[0]
    cv2.line(img, (east, 0), (east, height), (0, 0, 255), 10)
    cv2.putText(img, "E", (east, height), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 3)
    south = sphere2xy(pi, 0)[0]
    cv2.line(img, (south, 0), (south, height), (0, 0, 255), 10)
    cv2.putText(img, "S", (south, height), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 3)
    west = sphere2xy(3*pi/2, 0)[0]
    cv2.line(img, (west, 0), (west, height), (0, 0, 255), 10)
    cv2.putText(img, "W", (west, height), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 3)
    north = sphere2xy(0, 0)[0]
    cv2.line(img, (north, 0), (north, height), (0, 0, 255), 10)
    cv2.putText(img, "N", (north, height), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 3)
    # Draw the horizon
    cv2.line(img, (0, param["fullHeight"]//2 - param["ystart"]), (width, param["fullHeight"]//2 - param["ystart"]), (0, 0, 255), 5)

    I_ss = 0
    I_cs = 0
    I_c = 0
    for d in range(0, 365):
        st = time.time()
        # img_c = img.copy()
        for i in range(0, p_len):
            h = i * dt
            #tm = datetime.datetime(2022, 5, 27, h//60, h%60, 0, tzinfo=pytz.timezone("US/Eastern")).timetuple()
            a, z = get_sun_pos(d, h//60, h%60, 0, lat, long, -4)
            # print("{} -> day {} hour {}, Sun at position {}, {}\n".format(time.asctime(tm), tm.tm_yday, tm.tm_hour, a*180/pi, z*180/pi))
            positions[i, 0] = a
            positions[i, 1] = z

        """for i in range(0, len(positions)):
            p1 = sphere2xy(*positions[i])
            p2 = sphere2xy(*positions[(i+1)%p_len])
            if p1[0] > p2[0]: continue
            # print("line from {} to {}".format(p1, p2))
            cv2.line(img_c, p1, p2, (0, 255, 0), 5)"""
        t = 0
        for a, z in positions:
            pt = sphere2xy(a, z)
            if 0 <= pt[1] < height and 0 <= pt[0] < width and mask[2, pt[1]+1, pt[0]+1]:
                cv2.circle(img, pt, 10, (0, 0, 255), -1)
                dI_ss = (dt * 60) * sin(a) * sin(z) * 1353 * 0.7**((cos(z))**-0.678)
                dI_cs = (dt * 60) * cos(a) * sin(z) * 1353 * 0.7**((cos(z))**-0.678)
                dI_c = (dt * 60) * cos(z) * 1353 * 0.7**((cos(z))**-0.678)
                # print(a, z, dI_ss, dI_cs, dI_c)
                I_ss += dI_ss
                I_cs += dI_cs
                I_c += dI_c
            else:
                cv2.circle(img, pt, 10, (100, 100, 100), -1)

            if (t % (60*24/15)) == 4*18 + 3:
                cv2.circle(img, pt, 10, (0, 255, 255), -1)

            t += 1

    best_az = arctan(I_ss / I_cs) + pi
    best_ze = arctan((sin(best_az)*I_ss + cos(best_az)*I_cs) / (I_c))
    # without sin: 2361348871.1300774
    # with sin:    2349379502.0488343
    print("Best solar panel azimuth angle: {}".format(best_az * 180/pi))
    print("Best solar panel zenith angle:  {}".format(best_ze * 180/pi), arctan((I_ss**2 + I_cs**2)**0.5 / I_c) * 180/pi)
    print("Total energy:                   {} J/m^2".format(sin(best_az)*sin(best_ze)*I_ss + cos(best_az)*sin(best_ze)*I_cs + cos(best_ze)*I_c))
    pt = sphere2xy(best_az, best_ze)
    cv2.circle(img, pt, 50, (0, 255, 0), -1)
    cv2.circle(img, (pt[0], -pt[1]), 50, (0, 255, 255), -1)
    cv2.line(img, (pt[0], 0), (pt[0], height), (0, 255, 0), 5)

    cv2.destroyAllWindows()
    cv2.imwrite("/tmp/final.jpg", img)
    subprocess.call(["exiftool", "-TagsFromFile", sys.argv[1], '"-all:all>all:all"', "/tmp/final.jpg"])
    cv2.imshow("Processing", img)
    cv2.waitKey(0)
