import React from "react";
import { useState } from "react";
import axios from "axios";

import { Button, Typography, Box, Grid } from "@mui/material";
import ClassCard from "../components/ClassCard";

export function RandomPartyGeneratorPage() {
    const description = "This is the Random Party Generator page. It will be used to generate random parties for the user to use in their game.";

    const [classes, setClasses] = useState([]);
    const [loading, setLoading] = useState(false);

    function generateParty() {
        if (loading) {
            return;
        }
        setLoading(true);

        axios({
            url: 'http://localhost:5000/api/v1/generate_class',
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            data: {
                "num_characters": 4,
            }
        }).then((response) => {
            console.dir(response)
            setClasses(response.data);
        }).catch((error) => {
            console.error(error)
        }).finally(() => {
            setLoading(false);
        })
    }

    return (
        <div>
            <Typography variant="h2" align="center">Random Party Generator</Typography>
            <Typography variant="body1" align="center" sx={{padding: 2}}>{description}</Typography>
            <Grid container justifyContent='center' sx={{padding: 2}}>
                <Grid item>
                    <Button variant="contained" color="primary" onClick={() => {
                        generateParty();
                    }}>Generate Party</Button>
                </Grid>
            </Grid>
            <Grid container spacing={2} padding={5}>
                {classes.map((classItem) => {
                    return (
                        <Grid item xs={12} md={6} lg={3} key={classItem.name}>
                            <ClassCard name={classItem.name} description={classItem.description} classItems={classItem.classes} image={`http://localhost:5000/images/${classItem.image_id}`} />
                        </Grid>
                    );
                })}
            </Grid>
        </div>
    );
}