import {Label} from "react-bootstrap";
import React from "react";

const classes = {
    geo: ["success",    " (GEO)"],
    org: ["primary",    " (ORG)"],
    per: ["warning",    " (PER)"],
    gpe: ["info",       " (GPE)"],
    tim: ["primary",   " (TIME)"],
    art: ["danger",     " (ART)"],
    eve: ["default",  " (EVENT)"],
    nat: ["success", " (NATURE)"],
};

export function getLabelForClass(text, classification) {
    classification = classification.toLowerCase()
    for (const key in classes) {
        if (classification.includes(key))
            return <Label bsStyle={classes[key][0]}>{text + classes[key][1]}</Label>
    }
}