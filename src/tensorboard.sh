source /etc/profile
module load anaconda/2021a

PORTAL_FWNAME="$(id -un | tr '[A-Z]' '[a-z]')-tensorboard"
PORTAL_FWFILE="/home/gridsan/portal-url-fw/${PORTAL_FWNAME}"
rm $PORTAL_FWFILE
echo "http://$(hostname -s):${SLURM_STEP_RESV_PORTS}/" >> $PORTAL_FWFILE
chmod u+x ${PORTAL_FWFILE}

tensorboard --logdir $1 --host "$(hostname -s)" --port ${SLURM_STEP_RESV_PORTS}