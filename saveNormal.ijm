
run("Normalize Local Contrast", "block_radius_x=40 block_radius_y=40 standard_deviations=3 center stretch stack");
dir = getDirectory("image");
name = getTitle();
a = split(name, " ");
place = dir + "p"+a[1]+"f"+substring(a[4], 0, 1)+"_normal";
File.makeDirectory(place);
run("Image Sequence... ", "format=JPEG name=piece- start=1 save=" + place);