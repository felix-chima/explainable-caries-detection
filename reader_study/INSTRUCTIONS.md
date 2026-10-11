# Caries reader study: instructions

Thank you for taking part. This takes about 60 to 90 minutes.

## Task
For each of the 50 panoramic radiographs in `images/`, mark **every tooth with dental caries**
(any depth, including deep caries) with a rectangle. Draw the rectangle around the **whole
affected tooth** (crown and root), which is how the reference dataset was annotated.
If a radiograph has no caries, leave it without boxes.

Please work independently, do not discuss cases with other readers, and do not use any AI tool.
Use the screen and viewing conditions you would normally use.

## Tool: makesense.ai (free, runs in your browser, images are not uploaded to a server)
1. Open https://www.makesense.ai and click **Get Started**.
2. Drag in all 50 images from `images/` and choose **Object Detection**.
3. Create one label named exactly `caries`, then **Start project**.
4. Draw a rectangle around each carious tooth. Use the arrow keys to move between images.
5. When finished: **Actions > Export Annotations > "A .zip package containing files in YOLO format"**.
6. Send the zip back, renamed with your initials (for example `AB.zip`). Please also send:
   your role (dentist, specialist, student), years of experience, and the time taken.

Questions: contact the researcher.
