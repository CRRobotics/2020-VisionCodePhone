package com.example.opencvproject;

import org.opencv.core.*;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;
import org.opencv.imgproc.*;
// import org.opencv.highgui.*;

// import java.awt.*;
import java.io.IOException;
import java.util.*;
import java.util.List;
// import javax.swing.JSlider;

public class VisionProcessor {
    // static {
    //     System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    // }


    /*
     * Constants
     * NOTE: Check on these constants once the robot is assembled to ensure that they are still accurate
     * (e.g., the HEIGHT_OF_CAMERA may change through design)
     * NOTE: Some constants such as the FOV_X and FOV_Y, CAMERA_ANGLE, and FOCAL_DISTANCE  are not included.
     * These constants will be calibrated with sliders
     */

    public static final double WIDTH = 640.0;
    public static final double HEIGHT = 480.0;
    public static final double HEIGHT_OF_CAMERA = 38.5;
    public static final double HEIGHT_OF_TARGET = 8 * 12 + 2.25;
    public static final double HEIGHT_TO_TARGET = HEIGHT_OF_TARGET - HEIGHT_OF_CAMERA;
    public static final double OUTER_TARGET_WIDTH = 39.25;
    public static final double INNER_TARGET_DEPTH = 29.25;
    public static final double BALL_RADIUS = 3.5;
    public static final double SHOOTING_TOLERANCE = 1;
    public static final double MIN_AREA_CONTOUR = 500;
    public static final double CAMERA_ANGLE = 22 * Math.PI / 180 / 10;
    public static final double FOV_X = 54 * Math.PI / 180;

    public static final double FOCAL_DISTANCE = (WIDTH / 2) / Math.tan(FOV_X / 2);

    /*
     * A helper method that squares numbers, since Math.pow(number, 2) is slower
     * @return The input squared
     * @param a The number that will be squared
     */
    public static double pow(double a)
    {
        return a * a;
    }

    /*
     * Calculate the maximum robot angle relative to the inner target to be able to make an inner shot
     * Implement calculations in Fig. 15 of the documentation
     */
    public static final double MAX_ROBOT_ANGLE = Math.atan(OUTER_TARGET_WIDTH / (2 * INNER_TARGET_DEPTH)) - Math.atan((BALL_RADIUS + SHOOTING_TOLERANCE) / (Math.sqrt((INNER_TARGET_DEPTH * INNER_TARGET_DEPTH + pow(OUTER_TARGET_WIDTH / 2) - pow(BALL_RADIUS + SHOOTING_TOLERANCE)))));

    /*
     * @return The angle, in radians, from the line perpendicular to the inner target.
     * @precondition The distances must be the same units.
     * @param innerDistance The distance from the camera to the center point of the inner target
     * @param targetDistance The distance from the camera to the center of the outer target
     */
    public static double robot_angle(double innerDistance, double targetDistance, double calculatedInnerDepth)
    {
        return Math.acos((pow(innerDistance) + pow(calculatedInnerDepth) - pow(targetDistance)) / (2 * innerDistance * calculatedInnerDepth));
    }

    /*
     * @return the absolute angle, along a horizontal plane, to the center of the target, in radians
     * This function implements what is shown in Fig. 2 of the Documentation
     * relativeHorizontalAngle is the horizontal angle relative to the optical axis from the center of
     * the camera's fov to the center of the target, in radians (theta in Fig. 2)
     */
    public static double relative_horizontal_to_absolute_horizontal(double relativeHorizontalAngle)
    {
        return Math.atan(1 / Math.cos(CAMERA_ANGLE) * Math.tan(relativeHorizontalAngle));
    }

    /*
     * This function returns the absolute elevation angle, from horizontal to the
     * center of the target, in radians
     * This function implements what is shown in Fig. 1 of the Documentation
     * relativeElevationAngle is the elevation angle relative to the optical axis from the center of the
     * camera's fov to the center of the target (\varphi in Fig. 1)
     * relativeHorizontalAngle is the horizontal angle relative to the optical axis from the center of
     * the camera's fov to the center of the target, in radians (\theta in Fig. 1)
     */
    public static double relative_elevation_to_absolute_elevation(double relativeElevationAngle, double relativeHorizontalAngle)
    {
        return Math.asin(Math.cos(CAMERA_ANGLE) * Math.sin(relativeElevationAngle) + Math.cos(relativeElevationAngle) * Math.sin(CAMERA_ANGLE) * Math.cos(relativeHorizontalAngle));
    }

    public static String str(double num)
    {
        return num + "";
    }

    public static double round(double x, int n)
    {
        return (Math.round(x * Math.pow(10, n)) / (double) Math.pow(10, n));
    }

    //Grip Filters
    static final double[] hue                   = {0, 90};
    static final double[] saturation            = {60.0, 255.0};
    static final double[] value                 = {130.0, 255.0};
    static final Scalar lower                   = new Scalar(hue[0], saturation[0], value[0]);
    static final Scalar upper                   = new Scalar(hue[1], saturation[1], value[1]);

    public static void main(String[] args) {
        //JSlider minHue = new JSlider(JSlider.HORIZONTAL, 0, 255, INITIAL_MIN_HUE);
        System.out.print(System.getProperty("os.name"));
        if (!System.getProperty("os.name").contains("Windows")) {
            Runtime rt = Runtime.getRuntime();
            try {
                Process pr = rt.exec("sudo v4l2-ctl -c exposure_auto=1 -c exposure_absolute=" + 1);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public Mat processImg(Mat img)
    {
        // Convert to HSV
        Mat img2 = new Mat();
        img.copyTo(img, img2);
        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2HSV, 3);
        Core.inRange(img, lower, upper, img);

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(img, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        List<MatOfPoint> output = new ArrayList<MatOfPoint>();
        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area < MIN_AREA_CONTOUR)
                continue;
            output.add(contour);
        }
        contours = output;

        List<MatOfPoint> convexHulls = new ArrayList();
        try {
            for (MatOfPoint contour : contours) {
                MatOfInt hull = null;
                Imgproc.convexHull(contour, hull);
                convexHulls.add(new MatOfPoint(hull));
            }
        } catch (NullPointerException e) {
            return img;
        }


        try {
            // Find the left-most and right-most points in the contour
            // These points delimit the top segment of the target half-hexagon
            Point left = contours.get(0).toArray()[0];
            Point right = contours.get(0).toArray()[0];
            for (Point p : contours.get(0).toArray()) {
                if (p.x < left.x)
                    left = p;
                if (p.x > right.x)
                    right = p;
            }
            // Not sure this is useful...
            Imgproc.circle(img2, left, 1, new Scalar(255, 0, 0), 4);
            Imgproc.circle(img2, right, 1, new Scalar(255, 0, 0), 4);
            Imgproc.circle(img2, new Point((int) ((left.x + right.x) / 2), (int) ((left.y + right.y) / 2)), 1,
                    new Scalar(255, 0, 0), 4);

            // (x_1, y_1) is the left-most point relative to the center of the sensor
            // (x_2, y_2) is the right-most point relative to the center of the sensor
            double x_1 = left.x - WIDTH / 2;
            double x_2 = right.x - WIDTH / 2;
            double y_1 = HEIGHT / 2 - left.y;
            double y_2 = HEIGHT / 2 - right.y;

            // Calculate relative horizontal and elevation angles to the left and right sides of the outer target
            // Implement calculations of Fig. 9 of the documentation
            double relativeLeftElevationAngle = Math.atan(y_1 / (Math.sqrt(FOCAL_DISTANCE * FOCAL_DISTANCE + x_1 * x_1)));
            double relativeRightElevationAngle = Math.atan(y_2 / (Math.sqrt(FOCAL_DISTANCE * FOCAL_DISTANCE + x_2 * x_2)));


            double relativeLeftHorizontalAngle = Math.atan(x_1 / FOCAL_DISTANCE);
            double relativeRightHorizontalAngle = Math.atan(x_2 / FOCAL_DISTANCE);


            // Calculate absolute elevation angles to the left and right sides of the outer target
            // Implement calculations in Fig. 10 of the documentation
            double absoluteLeftElevationAngle = relative_elevation_to_absolute_elevation(relativeLeftElevationAngle, relativeLeftHorizontalAngle);
            double absoluteRightElevationAngle = relative_elevation_to_absolute_elevation(relativeRightElevationAngle, relativeRightHorizontalAngle);


            // Calculate absolute horizontal angles to the left and right sides of the outer target
            // Implement calculations in Fig. 11 of the documentation
            double absoluteLeftHorizontalAngle = relative_horizontal_to_absolute_horizontal(relativeLeftHorizontalAngle);
            double absoluteRightHorizontalAngle = relative_horizontal_to_absolute_horizontal(relativeRightHorizontalAngle);


            // Calculate the coordinates relative to the camera of the left, right, and center points of the outer target
            // Implement calculations in Fig. 12 of the documentation
            double lx = HEIGHT_TO_TARGET / Math.tan(absoluteLeftElevationAngle) * Math.cos(absoluteLeftHorizontalAngle);
            double rx = HEIGHT_TO_TARGET / Math.tan(absoluteRightElevationAngle) * Math.cos(absoluteRightHorizontalAngle);


            double ly = HEIGHT_TO_TARGET / Math.tan(absoluteLeftElevationAngle) * Math.sin(absoluteLeftHorizontalAngle);
            double ry = HEIGHT_TO_TARGET / Math.tan(absoluteRightElevationAngle) * Math.sin(absoluteRightHorizontalAngle);


            double ox = (lx + rx) / 2;
            double oy = (ly + ry) / 2;

            // Calculate the coordinates relative to the camera of the center of the inner target
            // Implement calculations in Fig. 13 of the documentation
            double ix = ox + INNER_TARGET_DEPTH / OUTER_TARGET_WIDTH * (ry - ly);
            double iy = oy + INNER_TARGET_DEPTH / OUTER_TARGET_WIDTH * (lx - rx);

            // Calculate the outer target width and inner target depth for testing purposes
            // Values should be 39.25 and 29.25, respectively
            double calculatedTargetWidth = Math.pow((Math.pow((lx - rx), 2) + Math.pow((ly - ry), 2)), (0.5));
            double calculatedInnerDepth = Math.pow((Math.pow((ox - ix), 2) + Math.pow((oy - iy), 2)), (0.5));

            // Calculate the absolute horizontal angles to center of the outer and inner targets
            // Implements calculations similar to Fig. 12 of the documentation
            double absoluteOuterHorizontalAngle = Math.atan(oy / ox);
            double absoluteInnerHorizontalAngle = Math.atan(iy / ix);


            // Calculate the absolute elevation angles to center of the outer and inner targets
            // Implements calculations similar to Fig. 12 of the documentation
            double absoluteOuterElevationAngle = Math.atan(HEIGHT_TO_TARGET / (Math.pow((ox * ox + oy * oy), 0.5)));
            double absoluteInnerElevationAngle = Math.atan(HEIGHT_TO_TARGET / (Math.pow((ix * ix + iy * iy), 0.5)));


            // Calculate the absolute elevation angles to center of the outer and inner targets
            // Implements calculations similar to Fig. 12 of the documentation
            double absoluteOuterHorizontalDistance = Math.pow(ox * ox + oy * oy, 0.5);
            double absoluteInnerHorizontalDistance = Math.pow(ix * ix + iy * iy, 0.5);


            // Calculate the angle of the robot relative the line perpendicular to the inner target
            double robotAngle = robot_angle(absoluteInnerHorizontalDistance, absoluteOuterHorizontalDistance, calculatedInnerDepth);

            // If the robotAngle is less than the max angle, tn inner target shot is possible
            boolean innerTargetPossible = robotAngle < MAX_ROBOT_ANGLE;
            // Display all calculated  angles and distances
            Imgproc.putText(img2, "Outer Horizontal Distance: " + str(round(absoluteOuterHorizontalDistance, 2)), new Point(5, 20), 3, 0.5, new Scalar(0, 0, 255), 1, Imgproc.LINE_AA);
            Imgproc.putText(img2, "Outer Horizontal Angle: " + str(round(absoluteOuterHorizontalAngle * 180 / Math.PI, 2)), new Point(5, 40), 3, 0.5, new Scalar(0, 0, 255), 1, Imgproc.LINE_AA);
            Imgproc.putText(img2, "Outer Elevation Angle: " + str(round(absoluteOuterElevationAngle * 180 / Math.PI, 2)), new Point(5, 60), 3, 0.5, new Scalar(0, 0, 255), 1, Imgproc.LINE_AA);

            Imgproc.putText(img2, "Inner Horizontal Distance: " + str(round(absoluteInnerHorizontalDistance, 2)), new Point(5, 90), 3, 0.5, new Scalar(0, 0, 255), 1, Imgproc.LINE_AA);
            Imgproc.putText(img2, "Inner Horizontal Angle: " + str(round(absoluteInnerHorizontalAngle * 180 / Math.PI, 2)), new Point(5, 110), 3, 0.5, new Scalar(0, 0, 255), 1, Imgproc.LINE_AA);
            Imgproc.putText(img2, "Inner Elevation Angle: " + str(round(absoluteInnerElevationAngle * 180 / Math.PI, 2)), new Point(5, 130), 3, 0.5, new Scalar(0, 0, 255), 1, Imgproc.LINE_AA);

            Imgproc.putText(img2, "Robot Angle: " + str(round(robotAngle * 180 / Math.PI, 2)), new Point(5, 160), 3, 0.5, new Scalar(0, 0, 255), 1, Imgproc.LINE_AA);
            Imgproc.putText(img2, "Max Robot Angle: " + str(round(MAX_ROBOT_ANGLE * 180 / Math.PI, 2)), new Point(5, 180), 3, 0.5, new Scalar(0, 0, 255), 1, Imgproc.LINE_AA);
            Imgproc.putText(img2, "Calculated Target Width: " + str(round(calculatedTargetWidth, 2)), new Point(5, 230), 3, 0.5, new Scalar(0, 0, 255), 1, Imgproc.LINE_AA);
            Imgproc.putText(img2, "Calculated Inner Depth: " + str(round(calculatedInnerDepth, 2)), new Point(5, 250), 3, 0.5, new Scalar(0, 0, 255), 1, Imgproc.LINE_AA);
            Imgproc.putText(img2, "Inner Target Possible: " + innerTargetPossible, new Point(5, 200), 3, 0.5, new Scalar(0, 0, 255), 1, Imgproc.LINE_AA);
            Imgproc.drawContours(img2, convexHulls, -1, new Scalar(0, 255, 0), 1);
            Imgproc.line(img2, new Point(320, 0), new Point(320, 480), new Scalar(255, 255, 255), 1);

            Imgproc.drawContours(img2, convexHulls, -1, new Scalar(0, 255, 0), 1);
            Imgproc.line(img2, new Point(320, 0), new Point(320, 480), new Scalar(255, 255, 255), 1);
            return img2;
        } catch (IndexOutOfBoundsException | NullPointerException e) {
            return img;
        }
    }
}
