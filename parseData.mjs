

export default function (data){
    for( let d of data){
        let label = d.label;
        let image = d.image;
        for(let i = 0; i < image.length; i++){
            image[i] = image[i] / 255;
        }
        d.label = label;
        d.image = image;
    }
}