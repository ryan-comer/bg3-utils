import React from 'react';
import { Card, CardMedia, CardContent, Typography, CardHeader, Grid, Box } from '@mui/material';

export default function ClassCard(props) {
  return (
    <Card>
      <CardHeader
        title={props.name}
      />
      <CardMedia
        component="img"
        height='400px'
        image={props.image}
        />
      <CardContent>
        {props.classItems.map((classItem) => {
            return (
                <Box style={{
                    display: 'flex',
                    justifyContent: 'start',
                    alignItems: 'center',

                }}>
                    <img 
                        style={{
                            height: '60px',
                        }}
                        src={classItem.image} alt={classItem.name} />
                    <Typography>
                        {classItem.name}
                    </Typography>
                </Box>
            )
        })}
        <hr/>
        <Typography variant='body2' color='text.secondary'>
            {props.description}
        </Typography>
      </CardContent>
    </Card>
  );
}